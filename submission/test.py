import gym
import copy
import argparse
from policy import *
from typing import Any, Dict, Iterable, List, Tuple

class DataStore:
    def __init__(self):
        self._data = None
        self._agent_names = None

    def __call__(self, **kwargs):
        try:
            self._data = copy.deepcopy(dict(**kwargs))
        except RecursionError:
            self._data = copy.copy(dict(**kwargs))

    @property
    def data(self):
        return self._data

    @property
    def agent_names(self):
        return self._agent_names

    @agent_names.setter
    def agent_names(self, names: Iterable[str]):
        self._agent_names = copy.deepcopy(names)


class CopyData(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_ids: List[str], datastore: DataStore):
        super(CopyData, self).__init__(env)
        self._datastore = datastore
        self._datastore.agent_names = agent_ids

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment and makes a copy of info. The info copy is a private attribute and
        cannot be acccessed from outside.
        Args:
            action (Dict[str, Any]): Action for each agent.
        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, done, and info, for each agent is returned.
        """
        obs, rewards, dones, infos = self.env.step(action)
        self._datastore(infos=infos, dones=dones)
        return obs, rewards, dones, infos

scenarios=[
        "1_to_2lane_left_turn_c",
        "1_to_2lane_left_turn_t",
        "3lane_merge_multi_agent",
        "3lane_merge_single_agent",
        "3lane_cruise_multi_agent",
        "3lane_cruise_single_agent",
        "3lane_cut_in",
        "3lane_overtake",
    ]

def main(args):
    env = gym.make('smarts.env:multi-scenario-v0', scenario=args.scenario, headless=not args.envision, sumo_headless=not args.sumo)

    # Make datastore.
    datastore = DataStore()
    # Make a copy of original info.
    env = CopyData(env, list(env.agent_specs.keys()), datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    wrappers = submitted_wrappers()
    for wrapper in wrappers:
        env = wrapper(env)

    eval_episodes = args.episodes
    policy = Policy()

    for _ in range(eval_episodes):
        observations = env.reset()
        dones = {"__all__": False}

        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)

            for agent_name, agent_info in infos.items():
                obs = agent_info["env_obs"]
                agent_done = dones[agent_name]

                if agent_done:
                    if obs.events.reached_goal:
                        print('goal')
                    elif (
                        len(obs.events.collisions) > 0
                        or obs.events.off_road
                        or obs.events.reached_max_episode_steps
                    ):
                        print(obs.events)
                    else:
                        raise Exception(f"Unsupported agent done reason. Events: {obs.events}.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--episodes', type=int, help='test episodes (default: 50)', default=50)
    parser.add_argument('--scenario', type=str, help='testing scenarios', default='1_to_2lane_left_turn_c')
    parser.add_argument('--envision', action='store_true', help='visualize in envision', default=False)
    parser.add_argument('--sumo', action='store_true', help='visualize in sumo', default=False)
    args = parser.parse_args()

    main(args)