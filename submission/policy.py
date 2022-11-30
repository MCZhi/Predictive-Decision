import gym
from pathlib import Path
from typing import Any, Dict
from predictor import *
from planner import *
from observation import observation_adapter

class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError

class FormatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observer = observation_adapter(num_neighbors=5)      

    def observation(self, obs):
        wrapped_obs = {}
        
        for agent_id, agent_obs in obs.items():
            wrapped_ob = self.observer(agent_obs)
            wrapped_obs[agent_id] = (agent_obs, wrapped_ob)

        self.observer.timestep += 1

        return wrapped_obs

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.observer.reset()

        return self.observation(observation)

class FormatAct(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cycle = 3

    def action(self, action, step):
        wrapped_act = {}
        
        for i, a in action.items():
            wrapped_act[i] = a[step]

        return wrapped_act

    def step(self, action):
        n_agent = len(action.keys())

        for t in range(self.cycle):
            act = self.action(action, t)
            observations, rewards, dones, infos = self.env.step(act)

            if len(observations.keys()) != n_agent:
                break 

            if dones['__all__']:
                break

        return observations, rewards, dones, infos

def submitted_wrappers():
    wrappers = [FormatObs, FormatAct]

    return wrappers

class Policy(BasePolicy):
    def __init__(self):
        model = Path(__file__).absolute().parents[0] / 'predictor_5000_0.6726.pth'
        self.predictor = Predictor()
        self.predictor.load_state_dict(torch.load(model, map_location='cpu'))
        self.predictor.eval()
        self.planner = Planner(self.predictor)

    def act(self, obs: Dict[str, Any]):
        wrapped_act = {}

        for agent_id, agent_obs in obs.items():
            actions = self.planner.plan(agent_obs)
            wrapped_act.update({agent_id: actions})

        return wrapped_act
