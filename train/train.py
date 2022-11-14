import glob
import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from torch import nn, optim
from collections import deque
from itertools import cycle
from predictor import Predictor
from train_utils import predictor_train_step

def main():
    offline_datasets = glob.glob(f'/SMARTS/competition/offline_dataset/*')
    len_scenarios = len(offline_datasets)
    steps = max(len_scenarios, 30000)
    offline_datasets = cycle(offline_datasets)

    # create training log
    loss, ade, fde = deque(maxlen=2000), deque(maxlen=2000), deque(maxlen=2000)

    with open("train_log.csv", 'w') as csv_file: 
        writer = csv.writer(csv_file) 
        writer.writerow(['step', 'loss', 'lr', 'ADE', 'FDE'])

    # create predictor, planner, optmizer
    learning_rate = 2e-4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device}, Steps: {steps}')   
    predictor = Predictor().to(device)  
    optimizer = optim.AdamW(predictor.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steps//10, gamma=0.8)

    # begin training
    for step in range(steps):
        scene = next(offline_datasets)
        print(f"Step:{step+1} Training on {scene}")
        pkl_files = glob.glob(scene+'/*.pkl')
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as reader:
                data = pickle.load(reader)

            try:
                # train step
                optimizer.zero_grad()
                step_loss, step_ade, step_fde = predictor_train_step(predictor, data, device)
                step_loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 5)
                optimizer.step()
        
                # log
                loss.append(step_loss.item())
                ade.append(step_ade.item())
                fde.append(step_fde.item())
                scheduler.step()
            except Exception as e: 
                print(e)
                continue

        with open("train_log.csv", 'a') as csv_file: 
            writer = csv.writer(csv_file) 
            print(f"Loss: {np.nanmean(loss)}, ADE: {np.nanmean(ade)}, FDE: {np.nanmean(fde)}\n")
            writer.writerow([step, np.nanmean(loss), optimizer.param_groups[0]['lr'], np.nanmean(ade), np.nanmean(fde)])

    # save model
    path = list(Path(__file__).absolute().parents)[1]
    torch.save(predictor.state_dict(), f'{path}/submission/predictor.pth')

if __name__ == "__main__":
    main()
