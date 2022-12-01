# Predictive-Decision-making

## Framework
We propose an interaction-aware predictor to forecast the neighboring agents' future trajectories around the ego vehicle conditioned on the ego vehicle's potential plans. A sampling-based planner will do collision checking and select the optimal trajectory considering the distance to the goal, ride comfort, and safety. The overall framework of our method is given below.

![Overview of our method](./docs/process.png)

## Results
Examples of our framework navigating in various scenarios are shown below.
### Left turn
<video muted controls width=500> <source src="./docs/left_turn_c.mp4"  type="video/mp4"> </video>

### Merge
<video muted controls width=500> <source src="./docs/merge.mp4"  type="video/mp4"> </video>

### Overtake
 <video muted controls width=500> <source src="./docs/overtake.mp4"  type="video/mp4"> </video>
 
## How to use
### Create a new Conda environment
```bash
conda create -n smarts python=3.8
```

### Install the SMARTS simulator
```bash
conda activate smarts
```

Install [SMARTS simulator](https://github.com/huawei-noah/SMARTS).
```bash
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"
```

### Install Pytorch
```bash
conda install pytorch==1.12.0 -c pytorch
```

### Training
Run `train.py`. Leave other arguments vacant to use the default setting.
```bash
python train.py --use_exploration --use_interaction
```

### Testing
Run `test.py`. You need specify the path to the trained predictor `--model_path`. You can aslo set `--envision_gui` to visualize the performance of the framework in envision or set `--sumo_gui` to visualize in sumo.
```bash
python test.py --model_path /training_log/Exp/model.pth
```
To visualize in Envision (some bugs exist in showing the road map), you need to manually start the envision server and then go to `http://localhost:8081/`.
```bash
scl envision start -p 8081
```

## Citation
If you find this repo to be useful in your research, please consider citing our work
```

```
