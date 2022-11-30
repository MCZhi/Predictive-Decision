import torch
from torch.nn import functional as F
import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
import matplotlib.pyplot as plt

# data process
def data_process(data, batch_size):
    env_input = {'ego_state': [], 'ego_map': [], 'neighbors_state': [], 'neighbors_map': []}
    batch_plan = []
    batch_ground_truth = []

    # process a mini batch
    while len(batch_plan) < batch_size:
        id = random.choice(list(data.keys())) # sample a episode
        episode_data = data[id]
        timesteps = list(episode_data.keys())
        ego_id = id.split(';')[-1]
        t = random.choice(timesteps[:len(timesteps)-15]) # sample a timestep

        neighbors = []
        neighbors_map = []
        neighbors_ground_truth = []
        agent_list = []
        current = episode_data[t][ego_id]['state'].copy()

        # add agents
        for agent in episode_data[t].keys():
            if agent == ego_id:
                ego = get_history(episode_data, agent, t)
                ego = traj_transform_to_ego_frame(ego, current)
                env_input['ego_state'].append(ego)
                plan = get_future(episode_data, agent, t)
                plan = traj_transform_to_ego_frame(plan, current)
                batch_plan.append(plan)
                map = episode_data[t][agent]['map'].copy()
                map = map_transform_to_ego_frame(map, current)
                env_input['ego_map'].append(map)

            elif episode_data[t][agent]['map'] is not None: 
                agent_list.append(agent)
                neighbor = get_history(episode_data, agent, t)
                neighbor = traj_transform_to_ego_frame(neighbor, current)
                neighbors.append(neighbor)
                gt = get_future(episode_data, agent, t)
                gt = traj_transform_to_ego_frame(gt, current)
                neighbors_ground_truth.append(gt)
                map = episode_data[t][agent]['map'].copy()
                map = map_transform_to_ego_frame(map, current)
                neighbors_map.append(map)
                
            else:
                continue

        # pad missing agents
        if len(agent_list) < 5:
            neighbors.extend([np.zeros(shape=(11, 5))] * (5-len(agent_list)))
            neighbors_map.extend([np.zeros(shape=(3, 51, 4))] * (5-len(agent_list)))
            neighbors_ground_truth.extend([np.zeros(shape=(30, 5))] * (5-len(agent_list)))
            
        # add to dict
        env_input['neighbors_state'].append(np.stack(neighbors))
        env_input['neighbors_map'].append(np.stack(neighbors_map))
        batch_ground_truth.append(np.stack(neighbors_ground_truth))

    for k, v in env_input.items():
        env_input[k] = np.stack(v)

    plan = np.stack(batch_plan)
    ground_truth = np.stack(batch_ground_truth)

    return env_input, plan, ground_truth

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
    env_input, plan, ground_truth = data_process(data, batch_size)
    env_input = {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in env_input.items()}
    plan = torch.as_tensor(plan).float().to(device)
    ground_truth = torch.as_tensor(ground_truth).float().to(device)
    prediction = model(env_input, plan)
    loss = calculate_loss(prediction, ground_truth)
    ade, fde = calculate_metrics(prediction, ground_truth)
 
    return loss, ade, fde

def calculate_loss(prediction, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, :, 0, None]
    regression_loss = F.smooth_l1_loss(prediction * valid, ground_truth[..., :3])
    
    '''
    if sequence:
        timesteps = list(agent_dir.keys())
        for i, t in enumerate(timesteps[1:], start=1):
            for j, a in enumerate(agent_dir[t]):
                last_t = timesteps[i-1]
                if a in agent_dir[last_t]:
                    k = agent_dir[last_t].index(a)                  
                    curr_pred = transform_to_world_postion(prediction[i, j], ego_pos[i])      
                    last_pred = transform_to_world_postion(prediction[i-1, k], ego_pos[i-1])
                    delta_t = t - last_t
                    consistency_loss += F.smooth_l1_loss(curr_pred[:30-delta_t], last_pred[delta_t:])
    '''
    loss = regression_loss

    return loss

def calculate_metrics(prediction, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, :, 0, None]
    prediction = prediction * valid
    prediction_error = torch.norm(prediction[:, :, :, :2] - ground_truth[:, :, :, :2], dim=-1)

    predictorADE = torch.mean(prediction_error, dim=-1)
    predictorADE = torch.masked_select(predictorADE, valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_error[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return predictorADE, predictorFDE

def transform_to_world_postion(traj, ego_current):
    o_x = ego_current[0]
    o_y = ego_current[1]
    o_theta = ego_current[2]
    x = traj[:, 0]
    y = traj[:, 1]
    
    world_x = x * torch.cos(o_theta) - y * torch.sin(o_theta) + o_x
    world_y = x * torch.sin(o_theta) + y * torch.cos(o_theta) + o_y
    traj = torch.stack([world_x, world_y], dim=-1)

    return traj