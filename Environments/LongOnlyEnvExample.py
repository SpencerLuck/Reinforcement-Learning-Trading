# This environment simulates buying and selling a physical asset like a coin or share.
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
import random as random

COST = 0.0007
CAPITAL = 100_000
NEG_MUL = 2


class DataLoader:
    """
    The class for getting data for assets. Does not provide scope for the inclusion
    of index related data, this is purely for traded asset data.
    """

    def __init__(self, asset_path, train_test_perc=0.8, training=True):
        self.asset_path = asset_path
        self.train_test_perc = train_test_perc
        self.training = training
        self.getData()

        self.scaler = StandardScaler()
        # fit scaler from the first column onwards. compute mean and std [10]
        if not training:
            self.scaler.fit(self.fit_data[:, 1:])
        else:
            self.scaler.fit(self.data[:, 1:])

    def getData(self):

        # Import data from csv
        df = pd.read_csv(self.asset_path)
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High',
                           'low': 'Low', 'volume': 'Volume'}, inplace=True)

        # Reward - Not included in Observation Space.
        df['rf'] = df['Close'].pct_change().shift(-1)

        # time transformations
        day = 60 * 60 * 24  # seconds in the day
        week = 60 * 60 * 24 * 7  # seconds in week
        df['Day_sin_value'] = np.sin(df['time'] * (2 * np.pi / day))
        df['Day_cos_value'] = np.cos(df['time'] * (2 * np.pi / day))
        df['Week_sin_value'] = np.sin(df['time'] * (2 * np.pi / week))
        df['Week_cos_value'] = np.cos(df['time'] * (2 * np.pi / week))

        # Returns and volume changes
        for i in [1, 2, 5, 10, 20, 40]:
            df[f'ret-{i}'] = df['Close'].pct_change(i)
            df[f'vol-{i}'] = df['Volume'].pct_change(i)

        # Volatility
        for i in [5, 10, 20, 40]:
            df[f'volt-{i}'] = np.log(1 + df["ret-1"]).rolling(i).std()

        # Techincal indicators
        for i in [1, 3, 5, 10, 20]:
            df[f'slope-{i}'] = ta.slope(close=df['Close'], length=i)

        for i in [14]:
            df[f'pgo-{i}'] = ta.pgo(close=df['Close'], high=df['High'], low=df[f'Low'], length=i)
        df.dropna(inplace=True)

        for i in [20]:
            df[f'natr-{i}'] = ta.natr(high=df['High'], low=df['Low'], close=df['Close'], length=i)
        df.dropna(inplace=True)


        # Filtering, filling NA values with interpolate, replacing and droppping nan/inf
        for c in df.columns:
            df[c].interpolate('linear', limit_direction='both', inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if self.training: # [11]
            df = df.iloc[:round(len(df) * self.train_test_perc)]
        else:
            fit_df = df.copy()
            fit_df = fit_df.iloc[:round(len(df) * self.train_test_perc)]
            df = df.iloc[round(len(df) * self.train_test_perc):]
            self.fit_data = np.array(fit_df.iloc[:, 6:])

        # creating self reference for the df
        self.frame = df
        # taking column 6 onwards, the model is not getting raw price
        # it is getting rf, time_transforms, price_transforms
        self.data = np.array(df.iloc[:, 6:])
        return

    def scaleData(self):
        """"
        fit_transform() is used on training data to scale data
        and learn the scaling parameters
        """
        self.scaled_data = self.scaler.fit_transform(self.data[:, 1:])
        return

    def __len__(self):
        """"
        A special method for the implementation of the len function
        """
        return len(self.data)

    def __getitem__(self, idx, col_idx=None):
        """"
        A special method that defines behavior for when item is accessed,
        using the notation self[key]
        """
        if col_idx is None:
            return self.data[idx]
        elif col_idx < len(list(self.data.columns)):
            return self.data[idx][col_idx]
        else:
            raise IndexError


class TradingEnv:
    """
    Long Only Trading Environment for trading a single asset.
    The Agent interacts with the environment class through the step() function.
    Action Space: {-1: Sell, 0: Do Nothing, 1: Buy}
    """

    def __init__(self, asset_data,
                 initial_money=CAPITAL, trans_cost=COST, store_flag=1, past_holding=0,
                 capital_frac=0.2, running_thresh=0.1, cap_thresh=0.3, episode_len=1000, random_ep=False,
                 inaction_interval=24):

        self.past_holding = past_holding  # [1]
        self.capital_frac = capital_frac  # Fraction of capital to invest each time.
        self.cap_thresh = cap_thresh
        self.running_thresh = running_thresh
        self.trans_cost = trans_cost

        self.asset_data = asset_data
        self.terminal_idx = len(self.asset_data) - 1  # end index
        self.scaler = self.asset_data.scaler

        self.initial_cap = initial_money
        self.equity = self.initial_cap
        self.running_capital = self.initial_cap
        self.asset_inv = self.past_holding

        self.pointer = 0
        self.random_ep = random_ep  # jump around timeseries randomly [2]
        self.episode_len = episode_len  # length of each episode after jump [2]
        self.next_return, self.current_state = 0, None
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.inaction_counter = 0
        self.inaction_interval = inaction_interval
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Close']
        self.done = False

        self.store_flag = store_flag
        if self.store_flag == 1:
            self.store = {"action_store": [],
                          "reward_store": [],
                          "equity": [],
                          "capital": [],
                          "port_ret": []}

    def reset(self):
        self.equity = self.initial_cap
        self.running_capital = self.initial_cap
        # setting starting inv to past holding
        self.asset_inv = self.past_holding
        self.terminal_idx = len(self.asset_data) - 1

        if self.random_ep:  # [3]
            rand_num = random.randint(1, 2)
            length = self.episode_len * rand_num
            self.pointer = random.randint(0, self.terminal_idx - length - 1)
            self.terminal_idx = self.pointer + length

        else:
            self.pointer = 0

        self.next_return, self.current_state = self.get_state(self.pointer)
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.inaction_counter = 0
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Close']
        self.done = False

        if self.store_flag == 1:  # [8]
            self.store = {"action_store": [],
                          "reward_store": [],
                          "equity": [],
                          "capital": [],
                          "port_ret": []}

        return self.current_state

    def step(self, action):
        self.current_act = action
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Close']
        self.current_reward = self.calculate_reward()
        # update prev action with current action
        self.prev_act = self.current_act
        self.pointer += 1
        self.next_return, self.current_state = self.get_state(self.pointer)
        self.done = self.check_terminal()

        if self.done:
            reward_offset = 0
            # last element / first element
            ret = (self.store['equity'][-1] / self.store['equity'][-0]) - 1
            if self.pointer < self.terminal_idx:
                reward_offset += -1 * max(0.5, 1 - self.pointer / self.terminal_idx)
            if self.store_flag == 1:
                reward_offset += 10 * ret
            self.current_reward += reward_offset

        if self.store_flag == 1:  # [8]
            self.store["action_store"].append(self.current_act)
            self.store["reward_store"].append(self.current_reward)
            self.store["equity"].append(self.equity)
            self.store["capital"].append(self.running_capital)
            info = self.store
        else:
            info = None

        return self.current_state, self.current_reward, self.done, info

    def calculate_reward(self):
        # amount to invest
        investment = self.running_capital * self.capital_frac
        reward_offset = 0
        act_value = 0

        # Buy Action [4]
        if self.current_act == 2:
            # check if still above the running thresh
            if self.running_capital > self.initial_cap * self.running_thresh:
                self.running_capital -= investment
                asset_units = investment / self.current_price
                self.asset_inv += asset_units
                # transaction cost factored into trade value
                self.current_price *= (1 - self.trans_cost)
                self.inaction_counter = 0
                act_value = 1

        # Sell Action [5]
        elif self.current_act == 1:
            # if we have holdings to sell
            if self.asset_inv > 0:
                self.running_capital += self.asset_inv * self.current_price * (1 - self.trans_cost)
                self.asset_inv = 0
                self.inaction_counter = 0
                act_value = -1

        # Do Nothing [6]
        elif self.current_act == 0:
            self.inaction_counter += 1
            # penalty for inaction
            reward_offset += -0.1 * self.inaction_counter / self.inaction_interval
            act_value = 0

        # Reward to give [7]
        prev_equity = self.equity
        self.equity = self.running_capital + self.asset_inv * self.current_price
        # if next_ret is -ve and current_act -ve then we profit. This forces action space -1 - +1.

        ## REWARD FUNCTION ##
        reward = 100 * self.next_return * act_value - np.abs(
            self.current_act - self.prev_act) * self.trans_cost

        if self.store_flag == 1:
            self.store['port_ret'].append((self.equity - prev_equity) / prev_equity)

        if reward < 0:
            reward *= NEG_MUL  # To make the Agent more risk averse towards negative returns.

        # updating rewards by reward offset
        reward += reward_offset

        return reward

    def check_terminal(self):
        # returning integers instead of booleans
        if self.pointer == self.terminal_idx:
            return 1
        elif self.equity <= self.initial_cap * self.cap_thresh:
            return 1
        else:
            return 0

    def get_state(self, idx):  # [9]
        # excluding the next return, everything else [4]
        state = self.asset_data[idx][1:]
        state = self.scaler.transform(state.reshape(1, -1))

        state = np.concatenate([state, [[self.equity / self.initial_cap,
                                         self.running_capital / self.equity,
                                         self.asset_inv * self.current_price / self.initial_cap,
                                         self.prev_act]]], axis=-1)
        # storing the next return
        next_ret = self.asset_data[idx][0]
        return next_ret, state

    def render(self, training=True):
        if training:
            print(f'TRAINING:'
                  f'Train_Capital_ret: {(self.running_capital - self.initial_cap) / self.initial_cap * 100}%, '
                  f'Train_Equity_ret: {(self.equity - self.initial_cap) / self.initial_cap * 100}%, '
                  f'Train_Asset_holding: {self.asset_inv}')
        else:
            print(f'VALIDATION:'
                  f'Val_Capital_ret: {(self.running_capital - self.initial_cap) / self.initial_cap * 100}%, '
                  f'Val_Equity_ret: {(self.equity - self.initial_cap) / self.initial_cap * 100}%, '
                  f'Val_Asset_holding: {self.asset_inv}')
