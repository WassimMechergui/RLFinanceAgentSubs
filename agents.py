from finrl.agents.stablebaselines3.models import DRLAgent
import numpy as np

class agent_wrapper:
  """
  A wrapper around the FinRL implementation of agents
  """

  def __init__(self, env):
    """
      Initialize the agent
    
    parameters :
      env : the used environment
    """

    self.env = env
    self.agent = DRLAgent(env = env)
    self.trained = False
    self.setup = False
    
  def setup_model(self, model_type, model_params):
    """
      Used to set up the model used to train the agent
    """
    self.model_ppo = self.agent.get_model("ppo",model_kwargs = model_params)
    self.setup = True

  def train(self, tb_log_name, total_timesteps =  80000):
    """
      Function used to train the agent based on the setup model.
    """
    assert self.setup, "model needs to be setup first."
    self.trained_ppo = self.agent.train_model(model=self.model_ppo, 
                                    tb_log_name=tb_log_name,
                                    total_timesteps=total_timesteps)
    self.trained = True

  def save(self, path):
    assert self.trained, "model needs to be trained first"
    self.trained_ppo.save(path)

  def predict(self, env):
    """
      Function to trade using the agent
    """
    return DRLAgent.DRL_prediction(model=self.trained_ppo, environment = env)





