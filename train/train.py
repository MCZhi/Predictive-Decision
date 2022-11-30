import random
import gym
import csv
import os
import torch
import numpy as np
import argparse
from torch import nn, optim
from itertools import cycle
from collections import deque
from observation import observation_adapter
from predictor import Predictor
from planner import Planner
from train_utils import predictor_train_step
from smarts.core.utils.episodes import episodes

def main(args):
    scenarios = [
        "1_to_2lane_left_turn_c",
        "1_to_2lane_left_turn_t",
        #"3lane_merge_multi_agent",
        "3lane_merge_single_agent",
        #"3lane_cruise_multi_agent",
        "3lane_cruise_single_agent",
        "3lane_cut_in",
        "3lane_overtake"
    ]

    scenarios_iter = cycle(scenarios)

    envs = {}
    for scen in scenarios:
        envs[f"{scen}"] = gym.make('smarts.env:multi-scenario-v0', scenario=scen)

    # create training log
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    loss, ade, fde = deque(maxlen=3000), deque(maxlen=3000), deque(maxlen=3000)
    success_rate = deque(np.zeros(60), maxlen=60)
    train_step = 0
    train_epoch = 0
    timesteps = 0

    with open(log_path+"predictor_log.csv", 'w') as csv_file: 
        writer = csv.writer(csv_file) 
        writer.writerow(['steps', 'loss', 'lr', 'ADE', 'FDE'])

    with open(log_path+"planner_log.csv", 'w') as csv_file: 
        writer = csv.writer(csv_file) 
        writer.writerow(['episodes', 'timesteps', 'collision', 'success', 'success_rate'])

    # create predictor, planner, optmizer, replay buffer
    predictor = Predictor().to(args.device)  
    planner = Planner(predictor)
    optimizer = optim.AdamW(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.9)
    replay_buffer = dict()

    # begin training
    for episode in episodes(args.epochs):
        scen = next(scenarios_iter)
        print(f"\nTraining on {scen}.\n")
        env = envs[scen]
        observer = observation_adapter(num_neighbors=5)
        observations = env.reset()
        observer.reset()
        episode.record_scenario(env.scenario_log)
        episode_len = 0
        planner.predictor.eval()

        # online interaction
        dones = {"__all__": False}
        while not dones["__all__"]:
            # plan
            env_input = observer(observations['Agent_0'])
            actions = planner.plan(observations['Agent_0'], env_input)

            # execute 3 steps
            for t in range(3):
                action = {'Agent_0': actions[t]}
                observations, rewards, dones, infos = env.step(action)
                timesteps += 1
                observer(observations['Agent_0'])
                episode.record_step(observations, rewards, dones, infos)
                episode_len += 1
                if dones["__all__"]:
                    break
        train_epoch += 1

        # train predictor
        if len(replay_buffer) >= 20:
            planner.predictor.train()

            for _ in range(20):
                # sample data and do gradient step
                optimizer.zero_grad()
                step_loss, step_ade, step_fde = predictor_train_step(predictor, replay_buffer, args.batch_size, args.device)
                step_loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 5)
                optimizer.step()
  
                # log
                loss.append(step_loss.item())
                ade.append(step_ade.item())
                fde.append(step_fde.item())
                scheduler.step()
                train_step += 1

        # dump episodic buffer to replay buffer
        episode_name = f'{scen}_{train_epoch}'
        if len(observer.buffer) > 30:
            replay_buffer[episode_name] = observer.buffer

        # purge some scenarios after certain episodes
        if len(replay_buffer) > 2000:
            s = random.choice(list(replay_buffer.keys()))   
            del replay_buffer[s]       

        # write to csv
        if train_step > 0:
            with open(log_path+"predictor_log.csv", 'a') as csv_file: 
                writer = csv.writer(csv_file) 
                print(f"Loss: {np.mean(loss)}, ADE: {np.mean(ade)}, FDE: {np.mean(fde)}")
                writer.writerow([train_step, np.mean(loss), optimizer.param_groups[0]['lr'], np.mean(ade), np.mean(fde)])
        
        if episode_len > 5:
            with open(log_path+"planner_log.csv", 'a') as csv_file: 
                writer = csv.writer(csv_file) 
                success = observations['Agent_0'].events.reached_goal
                collision = any(observations['Agent_0'].events.collisions)
                success_rate.append(1 if success else 0)
                print(f"Success: {success}, Collision: {collision}")
                writer.writerow([train_epoch, timesteps, collision, success, np.mean(success_rate)])

        # save model
        if (train_epoch+1) % 1000 == 0:
            torch.save(predictor.state_dict(), log_path+f'predictor_{train_epoch+1}_{np.mean(fde):.4f}.pth')

    # close all environments
    for env in envs.values():
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--epochs', type=int, help='training epochs (default: 5000)', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 64)', default=128)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    main(args)
