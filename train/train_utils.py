import torch
from torch.nn import functional as F
import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
import matplotlib.pyplot as plt

# data process
def data_process(data, batch_size):
    env_input = {'ego_state': [], 'neighbors_state': []}
    batch_plan = []
    batch_ground_truth = []

    # process a mini batch
    while len(batch_plan) < batch_size:
        e = random.choice(list(data.keys())) # sample an episode
        episode_data = data[e]
        timesteps = list(episode_data.keys())
        t = random.choice(timesteps[:len(timesteps)-15]) # sample a timestep
    
        neighbors = []
        neighbors_ground_truth = []
        agent_list = []
        ego_id = random.choice(list(episode_data[t].keys())) # sample an agent
        current = episode_data[t][ego_id]['state'].copy()    
        neighbors_dict = {}
        for k, v in episode_data[t].items():
            if k == ego_id:
                continue
            else:
                neighbors_dict[k] = v['state'][:2]

        sorted_neighbors = sorted(neighbors_dict.items(), key=lambda item: np.linalg.norm(item[1] - current[:2]))
        sorted_neighbors = sorted_neighbors[:5]
        neighbor_ids = [neighbor[0] for neighbor in sorted_neighbors]

        # add agents
        for k in episode_data[t].keys():
            if k == ego_id:
                ego = get_history(episode_data, k, t)
                ego = traj_transform_to_ego_frame(ego, current)
                env_input['ego_state'].append(ego)
                plan = get_future(episode_data, k, t)
                plan = traj_transform_to_ego_frame(plan, current)
                batch_plan.append(plan)          

            elif k in neighbor_ids: 
                agent_list.append(k)
                neighbor = get_history(episode_data, k, t)
                neighbor = traj_transform_to_ego_frame(neighbor, current)
                neighbors.append(neighbor)
                gt = get_future(episode_data, k, t)
                gt = traj_transform_to_ego_frame(gt, current)
                neighbors_ground_truth.append(gt)
      
            else:
                continue

        # pad missing agents
        if len(agent_list) < 5:
            neighbors.extend([np.zeros(shape=(11, 5))] * (5-len(agent_list)))
            neighbors_ground_truth.extend([np.zeros(shape=(30, 5))] * (5-len(agent_list)))
            
        # add to dict
        env_input['neighbors_state'].append(np.stack(neighbors))
        batch_ground_truth.append(np.stack(neighbors_ground_truth))

    for k, v in env_input.items():
        env_input[k] = np.stack(v)

    plan = np.stack(batch_plan)
    ground_truth = np.stack(batch_ground_truth)

    return env_input, ground_truth

def get_history(buffer, id, timestep):
    history = np.zeros(shape=(11, 4))
    timesteps = range(timestep+1)
    idx = -1

    for t in reversed(timesteps):
        if id not in buffer[t].keys() or idx < -11:
            break 

        history[idx] = buffer[t][id]['state'].copy()
        idx -= 1

    return history

def get_future(buffer, id, timestep):
    future = np.zeros(shape=(30, 4))
    timesteps = range(timestep+1, timestep+31)
 
    for idx, t in enumerate(timesteps):
        if id not in buffer[t].keys():
            break

        future[idx] = buffer[t][id]['state'].copy()

    return future

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

def traj_transform_to_ego_frame(traj, ego_current):
    line = LineString(traj[:, :2])
    center, angle = ego_current[:2], ego_current[2]
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[traj[:, :2]==0] = 0
    heading = wrap_to_pi(traj[:, 2] - angle)
    v = traj[:, 3]
    v_x, v_y = v * np.cos(heading), v * np.sin(heading)
    traj = np.column_stack((line_rotate, heading, v_x, v_y))

    return traj

def map_transform_to_ego_frame(map, ego_current):
    center, angle = ego_current[:2], ego_current[2]

    for i in range(map.shape[0]):
        if map[i, 0, 0] != 0:
            line = LineString(map[i, :, :2])
            line = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
            line = rotate(line, -angle, origin=(0, 0), use_radians=True)
            line = np.array(line.coords)
            line[map[i, :, :2]==0] = 0
            heading = wrap_to_pi(map[i, :, 2] - angle)
            speed_limit = map[i, :, 3]
            map[i] = np.column_stack((line, heading, speed_limit))

    return map

# train
def predictor_train_step(model, data, batch_size, device):
    env_input, ground_truth = data_process(data, batch_size)
    env_input = {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in env_input.items()}
    ground_truth = torch.as_tensor(ground_truth).float().to(device)
    prediction, score = model(env_input)
    loss, prediction = calculate_loss(prediction, score, ground_truth)
    ade, fde = calculate_metrics(prediction, ground_truth)
 
    return loss, ade, fde

def calculate_loss(prediction, score, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, None, :, 0, None]
    trajectories = prediction * valid
    distance = torch.norm(trajectories[:, :, :, 4::5, :2] - ground_truth[:, :, None, 4::5, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode.unsqueeze(-1)]
    best_mode_future = best_mode_future.squeeze(2)
    loss = F.smooth_l1_loss(best_mode_future, ground_truth[..., :3])
    loss += 0.5 * F.cross_entropy(score.permute(0, 2, 1), best_mode)

    return loss, best_mode_future

def calculate_metrics(prediction, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, :, 0]
    prediction = prediction * valid.unsqueeze(-1)
    prediction_error = torch.norm(prediction[:, :, :, :2] - ground_truth[:, :, :, :2], dim=-1)

    predictorADE = torch.mean(prediction_error, dim=-1)
    predictorADE = torch.masked_select(predictorADE, valid[:, :, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_error[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, valid[:, :, 0])
    predictorFDE = torch.mean(predictorFDE)

    return predictorADE, predictorFDE

