import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from smarts.core.utils.math import position_to_ego_frame, wrap_value
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import Waypoints, RoadWaypoints

class observation_adapter(object):
    def __init__(self, env, num_neighbors=5):
        self.num_neighbors = num_neighbors        
        self.hist_steps = 11
        self.num_lanes = 3
        self.num_waypoints = 51
        self._env = env.unwrapped._smarts

        # neighbor vehicle map spec
        self.neighbor_interface = AgentInterface(
            waypoints=Waypoints(lookahead=self.num_waypoints-1),
            road_waypoints=RoadWaypoints(horizon=self.num_waypoints-1),
        )

    def cache(self, env_obs):
        ego = env_obs.ego_vehicle_state
        self.ego_id = ego.id
        self.buffer[self.timestep][ego.id] = {'state': np.concatenate([ego.position[:2], [ego.heading+np.pi/2, ego.speed]]), 'map': None}
        
        neighbors = env_obs.neighborhood_vehicle_states
        for neighbor in neighbors:
            self.buffer[self.timestep][neighbor.id] = {'state': np.concatenate([neighbor.position[:2], [neighbor.heading+np.pi/2, neighbor.speed]]), 'map': None}

    def reset(self):
        self.buffer = defaultdict(dict)
        self.timestep = 0

    def ego_history_process(self, id):
        ego_history = np.zeros(shape=(self.hist_steps, 5))
        timesteps = list(self.buffer.keys())
        idx = -1

        for t in reversed(timesteps):
            pos = self.buffer[t][id]['state'][:2]
            head = self.buffer[t][id]['state'][2]
            speed = self.buffer[t][id]['state'][3]
            ego_history[idx, :2] = self.transform(np.append(pos, [0]))[:2]
            ego_history[idx, 2] = self.adjust_heading(head)
            ego_history[idx, 3:] = np.array((speed * np.cos(self.adjust_heading(head)), speed * np.sin(self.adjust_heading(head))))
            idx -= 1
            if idx < -self.hist_steps:
                break

        return ego_history
    
    def neighbor_history_process(self, ids):
        neighbor_history = np.zeros(shape=(self.num_neighbors, self.hist_steps, 5))

        for i, id in enumerate(ids):
            timesteps = list(self.buffer.keys())
            idx = -1

            for t in reversed(timesteps):
                if id not in self.buffer[t] or idx < -self.hist_steps:
                    break 
                
                pos = self.buffer[t][id]['state'][:2]
                head = self.buffer[t][id]['state'][2]
                speed = self.buffer[t][id]['state'][3]
                neighbor_history[i, idx, :2] = self.transform(np.append(pos, [0]))[:2]
                neighbor_history[i, idx, 2] = self.adjust_heading(head) 
                neighbor_history[i, idx, 3:] = np.array((speed * np.cos(self.adjust_heading(head)), speed * np.sin(self.adjust_heading(head))))
                idx -= 1

        return neighbor_history

    def ego_map_process(self, paths):
        ego_map = np.zeros(shape=(self.num_lanes, self.num_waypoints, 4))

        for i, path in enumerate(paths):
            if i >= self.num_lanes:
                break
            for j, point in enumerate(path):
                ego_map[i, j, :2] = self.transform(np.append(point.pos, [0]))[:2]
                ego_map[i, j, 2] = self.adjust_heading(point.heading+np.pi/2) 
                ego_map[i, j, 3] = point.speed_limit

        return ego_map

    def neighbor_map_process(self, neighbors):
        neighbors_map = np.zeros(shape=(self.num_neighbors, self.num_lanes, self.num_waypoints, 4))

        for idx, neighbor in enumerate(neighbors.values()):
            paths = neighbor.waypoint_paths
            for i, path in enumerate(paths):
                if i >= self.num_lanes:
                    break
                for j, point in enumerate(path):
                    neighbors_map[idx, i, j, :2] = self.transform(np.append(point.pos, [0]))[:2]
                    neighbors_map[idx, i, j, 2] = self.adjust_heading(point.heading+np.pi/2)
                    neighbors_map[idx, i, j, 3] = point.speed_limit

        return neighbors_map

    def __call__(self, env_obs):
        self.current_pos = (env_obs.ego_vehicle_state.position, env_obs.ego_vehicle_state.heading+np.pi/2)
        self.cache(env_obs)
        ego_state = self.ego_history_process(env_obs.ego_vehicle_state.id)

        # cache ego map waypoints
        map_waypoints = np.zeros(shape=(self.num_lanes, self.num_waypoints, 4))
        paths = env_obs.waypoint_paths

        for i, path in enumerate(paths):
            if i >= self.num_lanes:
                break
            for j, point in enumerate(path):
                map_waypoints[i, j, :2] = point.pos
                map_waypoints[i, j, 2] = point.heading+np.pi/2
                map_waypoints[i, j, 3] = point.speed_limit

        self.buffer[self.timestep][env_obs.ego_vehicle_state.id]['map'] = map_waypoints
        ego_map = self.ego_map_process(env_obs.waypoint_paths)
    
        neighbors = {}
        for neighbor in env_obs.neighborhood_vehicle_states:
            neighbors[neighbor.id] = neighbor.position[:2]
        
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: np.linalg.norm(item[1] - self.current_pos[0][:2]))
        sorted_neighbors = sorted_neighbors[:self.num_neighbors]
        neighbor_ids = [neighbor[0] for neighbor in sorted_neighbors]

        self._env.attach_sensors_to_vehicles(self.neighbor_interface, neighbor_ids)
        neighbors_obs, _, _, _ = self._env.observe_from(neighbor_ids)
        neighbors_state = self.neighbor_history_process(neighbor_ids)

        # cache neighbor map waypoints
        for id in neighbor_ids:
            map_waypoints = np.zeros(shape=(self.num_lanes, self.num_waypoints, 4))
            paths = neighbors_obs['Agent-'+id].waypoint_paths

            for i, path in enumerate(paths):
                if i >= self.num_lanes:
                    break
                for j, point in enumerate(path):
                    map_waypoints[i, j, :2] = point.pos
                    map_waypoints[i, j, 2] = point.heading+np.pi/2
                    map_waypoints[i, j, 3] = point.speed_limit

            self.buffer[self.timestep][id]['map'] = map_waypoints

        neighbors_map = self.neighbor_map_process(neighbors_obs)
        
        self.obs = {'ego_state': ego_state, 'ego_map': ego_map, 'neighbors_state': neighbors_state, 'neighbors_map': neighbors_map}
        self.timestep += 1

        return self.obs

    def transform(self, v):
        return position_to_ego_frame(v, self.current_pos[0], self.current_pos[1])

    def adjust_heading(self, h):
        return wrap_value(h - self.current_pos[1], -math.pi, math.pi)

    def ego_frame_dynamics(self, v):
        ego_v = v.copy()
        ego_v[0] = v[0] * np.cos(self.current_pos[1]) + v[1] * np.sin(self.current_pos[1])
        ego_v[1] = v[1] * np.cos(self.current_pos[1]) - v[0] * np.sin(self.current_pos[1])

        return ego_v

    def render(self):
        # plot agent
        ego = plt.Circle((self.obs['ego_state'][-1, 0], self.obs['ego_state'][-1, 1]), 0.6, color='r')
        plt.gca().add_patch(ego)
        plt.plot(self.obs['ego_state'][:, 0], self.obs['ego_state'][:, 1], 'r')

        for i in range(self.num_neighbors):
            if self.obs['neighbors_state'][i][-1][0] != 0:
                neighbor = plt.Circle((self.obs['neighbors_state'][i, -1, 0], self.obs['neighbors_state'][i, -1, 1]), 0.6, color='c')
                plt.gca().add_patch(neighbor) 
                plt.plot(self.obs['neighbors_state'][i, :, 0], self.obs['neighbors_state'][i, :, 1], 'c')

        # plot map
        for i in range(self.obs['ego_map'].shape[0]):   
            if self.obs['ego_map'][i, 0, 0] != 0:
                plt.plot(self.obs['ego_map'][i, :, 0], self.obs['ego_map'][i, :, 1], 'k--')

        for i in range(self.num_neighbors):
            for j in range(self.obs['neighbors_map'].shape[1]):
                if self.obs['neighbors_map'][i, j, 0, 0] != 0:
                    plt.plot(self.obs['neighbors_map'][i, j, :, 0], self.obs['neighbors_map'][i, j, :, 1], 'k--')

        # show
        plt.gca().set_aspect('equal')
        plt.show()