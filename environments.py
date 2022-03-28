import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

def softmax(arr):
    arr = np.array(arr)
    return np.exp(arr) / np.sum(np.exp(arr))

class StockPortfolioEnv(gym.Env):
    """
    (FR)Classe de modelisation du marchier financier d'échange de stock.
    A class to model the trading enviromnent for stocks. Modeled after the stock market. 

    Attributes
    ----------
        df: DataFrame
            input data containing the historical data
        stock_dim : int
            number of unique stocks
        initial_amount : int
            starting amount
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        sends to the broker (the environment) the sell action (based on the sign of the action)
    _buy_stock()
        sends to the broker (the environment) the buy action (based on the sign of the action)
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                initial_amount,
                state_space,
                action_space,
                tech_indicator_list,
                day = 0):
        """Initializes the environment.

        Args:
            df(DataFrame) : input data containing the historical data
            stock_dim(int) :  number of unique stocks
            initial_amount (int) : starting amount
            state_space (int) : the dimension of input features
            action_space (int) : equals stock dimension
            tech_indicator_list (list) : a list of technical indicator names
            day (int) : an increment number to control date
        """
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
        # Ex. For the DOW 30 the shape should be : (34, 30), In our case it'll be (34, 28)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))

        # Setup the state from the given day initial day index
        self.setup_state(self.day)
        self.terminal = False          
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount
        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        """Advances the environment by one step.

        Args:
            actions : action taken by the agent towards the environment

        Returns:
            tuple : Tuple containing several informations about the taken step : state / reward / ending step reach / info 
        """
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()
            
            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal, {}

        else:
            weights = softmax(actions) 
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.setup_state(self.day)
            
            # New portfolio value can the calculated from the variations of each stock
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value 
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        """Resets the environment to inital state and returns the initial observation.

        Returns:
            array : The initial observation
        """
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.setup_state(self.day)
        self.portfolio_value = self.initial_amount        
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]] 
        return self.state

    def setup_state(self, day_index):
        """Utility function to set up the state of the environment for a given day index.

        Args:
            day_index (int): The index of the considered day in the DataFrame.
        """
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)

    def render(self):
        """Rendering function. At this state just gives the state
        """
        return self.state
        
    def save_asset_memory(self):
        """Utility function.
        """
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """Utility function.
        """
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs