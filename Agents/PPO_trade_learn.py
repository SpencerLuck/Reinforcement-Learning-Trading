from PPO_torch_trade import Agent
from PerpEnv import DataLoader, TradingEnv
import warnings
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = 'ETHPERP_1hr.csv'  # Crypto data with funding rates as column
MODEL_NUM = 1
PAIR = 'ETHPERP'
MODEL_PATH = f'models\\ppo_torch\\model_{MODEL_NUM}_{PAIR}'


if not os.path.isdir(MODEL_PATH):
    mode = 0o666
    model_folder = f'model_{MODEL_NUM}_{PAIR}'
    m_path = os.path.join('models\\ppo_torch', model_folder)
    os.makedirs(m_path, mode)


def plot_learning_curve(x, scores, val=False):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    if val:
        plt.title('Validation running average of previous 100 scores')
        plt.savefig(f'plots\\Model torch {MODEL_NUM}_{PAIR} validation.png')
    else:
        plt.title('Running average of previous 100 scores')
        plt.savefig(f'plots\\Model torch {MODEL_NUM}_{PAIR}.png')


warnings.filterwarnings('ignore')
if __name__ == '__main__':

    N = 20
    BATCH_SIZE = 5
    N_EPOCHS = 4
    ACTOR_ALPHA = 0.0001
    CRITIC_ALPHA = 0.0003
    EPISODES = 3000
    N_ACTIONS = 4
    EPISODE_LEN = 2000
    INACTION_INTERVAL = 1
    POLICY_CLIP = 0.2


    # Load data
    train_data = DataLoader(asset_path=DATA_PATH, training=True, train_test_perc=0.8)
    val_data = DataLoader(asset_path=DATA_PATH, training=False, train_test_perc=0.8)

    # Load env
    train_env = TradingEnv(asset_data=train_data, random_ep=True, episode_len=EPISODE_LEN,
                           inaction_interval=INACTION_INTERVAL)
    val_env = TradingEnv(asset_data=val_data, random_ep=False)



    agent = Agent(n_actions=N_ACTIONS, batch_size=BATCH_SIZE,
                  actor_alpha=ACTOR_ALPHA, critic_alpha=CRITIC_ALPHA, n_epochs=N_EPOCHS,
                  input_dims=[1654], policy_clip=POLICY_CLIP)

    #     agent.load_models(file_path=MODEL_PATH)


    best_score = -100
    score_history = []
    val_score_history = []
    learn_iters = 0
    avg_score = 0
    avg_val_score = 0
    n_steps = 0

    for i in range(EPISODES):
        print('-' * 50)
        print(f'Episode {i}')
        observation = train_env.reset()
        done = False
        score = 0
        actions = []
        val_done = False
        val_observation = val_env.reset()
        val_score = 0

        # Training
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = train_env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

            if done:
                train_env.render(training=True)


        # Validation
        while not val_done:
            # TODO: modify choose action to better rep live trading
            val_action, val_prob, val_val = agent.choose_action(val_observation)
            val_observation_, val_reward, val_done, val_info = val_env.step(val_action)
            val_score += val_reward
            val_observation = val_observation_

            if val_done:
                val_env.render(training=False)

        # Collecting scores
        score_history.append(score)
        val_score_history.append(val_score)

        avg_score = np.mean(score_history[-100:])
        avg_val_score = np.mean(val_score_history[-100:])


        # Saving model if condition is met
        if avg_score > best_score:
            best_score = avg_score
            save_path = os.path.join(MODEL_PATH, f'episode_{i}')
            mode = 0o666
            os.makedirs(save_path, mode)
            agent.save_models(file_path=save_path, episode=i)

        # Displaying end of episode scores
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'val score %.1f' % val_score, 'avg val score %.1f' % avg_val_score)

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, val=False)

    y = [i + 1 for i in range(len(val_score_history))]
    plot_learning_curve(y, score_history, val=True)


