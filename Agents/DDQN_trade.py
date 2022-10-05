import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

T.manual_seed(0)
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, name, input_dims):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)
        self.name = name
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        state = state.float()
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        l3 = F.relu(self.fc3(l2))
        l4 = F.relu(self.fc4(l3))
        l5 = F.relu(self.fc5(l4))
        V = self.V(l5)
        A = self.A(l5)

        return V, A

    def save_checkpoint(self, file_path, episode):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(file_path, self.name + f'_dqn_{episode}')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(file_path))


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=5000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        # Action space
        self.action_space = [0, 1, 2, 3]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DuelingLinearDeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_eval')
        self.q_next = DuelingLinearDeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, training=True):
        if training:
            if np.random.random() > self.epsilon:
                observation = observation[np.newaxis,:]
                state = T.tensor(observation).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            observation = observation[np.newaxis, :]
            state = T.tensor(observation).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self, ep, ep_steps):
        step = self.epsilon_cycler(num_cycles=5, cycle_window=1000)
        c1 = 0
        c2 = c1 + step
        c3 = c2 + step

        if (ep==c1 or ep==c2 or ep==c3) and ep_steps==0:
            self.epsilon = 1
            self.epsilon = self.epsilon - self.eps_dec \
                             if self.epsilon > self.eps_min else self.eps_min
        else:
            self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

    @ staticmethod
    def epsilon_cycler(num_cycles, cycle_window):
        cycle_step = cycle_window / num_cycles
        return int(round(cycle_step))

    def learn(self, ep, ep_steps):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        # using T.Tensor seems to reset datatype to float
        # using T.tensor preserves source data type
        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state)
        V_s_, A_s_ = self.q_next.forward(new_state)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1,
                                              action.unsqueeze(-1)).squeeze(-1)

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_target = rewards + self.gamma*T.max(q_next, dim=1)[0].detach()
        q_target[dones] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon(ep, ep_steps)

    def save_models(self, file_path, episode):
        '''
        To save a model, specify the file path
        '''
        self.q_eval.save_checkpoint(file_path=file_path, episode=episode)
        self.q_next.save_checkpoint(file_path=file_path, episode=episode)


    def load_models(self, q_eval_file_path, q_next_file_path):
        '''
        To load a model, specify the file path for the model
        '''
        self.q_eval.load_checkpoint(file_path=q_eval_file_path)
        self.q_next.load_checkpoint(file_path=q_next_file_path)