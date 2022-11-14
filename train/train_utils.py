import torch
import random
import numpy as np
from torch.nn import functional as F
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from collections import defaultdict

# data process
def data_process(data):
    env_input = {'ego_state': [], 'neighbors_state': []}
    batch_plan = []
    batch_ground_truth = []

    # dump to buffer
    episode_data, ego_id = get_buffer(data)
    timesteps = list(episode_data.keys())

    # process a mini batch
    for t in timesteps[:len(timesteps)-15]:
        neighbors = []
        neighbors_ground_truth = []
        agent_list = []
        current = episode_data[t][ego_id].copy()    
        neighbors_dict = {}
        for k, v in episode_data[t].items():
            if k == ego_id:
                continue
            else:
                neighbors_dict[k] = v[:2]

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

def get_buffer(data):
    buffer = defaultdict(dict)

    for t, obs in data.items():
        ego = obs.ego_vehicle_state
        buffer[int(t*10)][ego.id] = np.concatenate([ego.position[:2], [ego.heading+np.pi/2, ego.speed]]).astype(np.float32)
        neighbors = obs.neighborhood_vehicle_states

        if neighbors is not None:
            for neighbor in neighbors:
                buffer[int(t*10)][neighbor.id] = np.concatenate([neighbor.position[:2], [neighbor.heading+np.pi/2, neighbor.speed]]).astype(np.float32)

    return buffer, ego.id

def get_history(buffer, id, timestep):
    history = np.zeros(shape=(11, 4))
    timesteps = range(timestep+1)
    idx = -1

    for t in reversed(timesteps):
        if id not in buffer[t].keys() or idx < -11:
            break 

        history[idx] = buffer[t][id].copy()
        idx -= 1

    return history

def get_future(buffer, id, timestep):
    future = np.zeros(shape=(30, 4))
    timesteps = range(timestep+1, timestep+31)
 
    for idx, t in enumerate(timesteps):
        if id not in buffer[t].keys():
            break

        future[idx] = buffer[t][id].copy()

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

# train
def predictor_train_step(model, data, device):
    env_input, ground_truth = data_process(data)
    env_input = {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in env_input.items()}
    ground_truth = torch.as_tensor(ground_truth).float().to(device)
    prediction, score = model(env_input)
    loss, prediction = calculate_loss(prediction, score, ground_truth)
    ade, fde = calculate_metrics(prediction, ground_truth)
 
    return loss, ade, fde

def calculate_loss(prediction, score, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, None, :, 0, None]
    trajectories = prediction * valid
    distance = torch.norm(trajectories[:, :, :, 4::5] - ground_truth[:, :, None, 4::5, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode.unsqueeze(-1)]
    best_mode_future = best_mode_future.squeeze(2)
    loss = F.smooth_l1_loss(best_mode_future, ground_truth[..., :2])
    loss += F.cross_entropy(score.permute(0, 2, 1), best_mode)

    return loss, best_mode_future

def calculate_metrics(prediction, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, :, 0]
    prediction = prediction * valid.unsqueeze(-1)
    prediction_error = torch.norm(prediction[:, :, :, :2] - ground_truth[:, :, :, :2], dim=-1)
    
    valid[0, 0, 0] = True # prevent nan
    predictorADE = torch.mean(prediction_error, dim=-1)
    predictorADE = torch.masked_select(predictorADE, valid[:, :, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_error[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, valid[:, :, 0])
    predictorFDE = torch.mean(predictorFDE)

    return predictorADE, predictorFDE