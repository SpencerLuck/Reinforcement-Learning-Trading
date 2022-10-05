import numpy as np
from DDQN_trade import Agent
from SpotEnv import DataLoader, TradingEnv
import warnings
import matplotlib.pyplot as plt
import os


DATA_PATH = 'ETHPERP_1hr.csv'  # data with future rates as a column
MODEL_NUM = 9
PAIR = 'ETHUSD'
MODEL_PATH = f'\\ReinforcementLearning\\models\\dqn\\model_{MODEL_NUM}_{PAIR}'


if not os.path.isdir(MODEL_PATH):
    mode = 0o666
    model_folder = f'model_{MODEL_NUM}_{PAIR}'
    m_path = os.path.join('\\ReinforcementLearning\\models\\dqn', model_folder)
    os.makedirs(m_path, mode)


def plot(train_scores, val_scores, episode, model_num, show=False):
    x = [i for i in range(1, len(val_scores) + 1)]
    y1 = train_scores
    y2 = val_scores
    plt.plot(x, y1, label='train_scores')
    plt.plot(x, y2, label='val_scores')
    plt.legend()
    if show:
        plt.show()
    plt.title(f'Model {model_num}_{PAIR} for ep {episode}')
    plt.savefig(f'plots\\Model {model_num}_{PAIR} for ep {episode}.png')


if __name__ == '__main__':

    # Load data
    train_data = DataLoader(asset_path=DATA_PATH, training=True, train_test_perc=0.8)
    train_env = TradingEnv(asset_data=train_data, random_ep=True, episode_len=2000, inaction_interval=24)

    val_data = DataLoader(asset_path=DATA_PATH, training=False, train_test_perc=0.8)
    val_env = TradingEnv(asset_data=val_data, random_ep=False)


    warnings.filterwarnings('ignore')

    episodes = 5_000
    load_checkpoint = False


    agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4,
                  input_dims=[1804], n_actions=4, mem_size=15_000, eps_min=0.01,
                  batch_size=256, eps_dec=1e-4, replace=1_000)


    if load_checkpoint:
        pass
        # agent.load_models(file_path=MODEL_PATH)

    scores = []
    val_scores = []

    for i in range(episodes):
        done = False
        observation = train_env.reset()
        score = 0
        ep_steps = 0

        val_done = False
        val_observation = val_env.reset()
        val_score = 0

        # Training
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = train_env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn(ep=i, ep_steps=ep_steps)
            ep_steps += 1

            observation = observation_

            if done:
                train_env.render(training=True)


        if i > 0 and i % 10 == 0:
            save_path = os.path.join(MODEL_PATH, f'episode_{i}')
            mode = 0o666
            os.makedirs(save_path, mode)
            agent.save_models(file_path=save_path, episode=i)

        # Validation
        while not val_done:
            val_action = agent.choose_action(val_observation, training=False)
            val_observation_, val_reward, val_done, val_info = val_env.step(val_action)
            val_score += val_reward
            val_observation = val_observation_

            if val_done:
                val_env.render(training=False)


        scores.append(score)
        val_scores.append(val_score)

        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        avg_val_score = np.mean(val_scores[max(0, i - 100):(i + 1)])
        print('episode: ', i, 'score %.1f ' % score,
              ' average score %.1f' % avg_score,
              ' average val score %.1f' % avg_val_score,
              ' epsilon %.2f' % agent.epsilon,
              ' ep steps %.2f' % ep_steps)

        if i > 0 and i % 100 == 0:
            plot(train_scores=scores, val_scores=val_scores,
                 episode=i, model_num=MODEL_NUM, show=False)

