# 2022 NeurIPS Driving SMARTS Competition Track 1 Solution
InterSim is a simulator for closed-loop interactive simulations with real on-road data and maps. Without over-promising, the best use case for InterSim Beta is to test your planning system before deployment extensively. If you want to test your planner with a large real-world dataset, you should consider InterSim. InterSim models real-world drivers' behaviors against new conflicts with relation prediction models. InterSim now supports both Waymo Open Motion Dataset and the NuPlan dataset.

# How to use

InterSim runs on pure Python with NO C/C++, NO ROS, and NO CARLA. InterSim is designed to be light and easy to use for all researchers.

## Create a new Conda environment
```bash
conda create -n smarts python=3.8
```

## Install the SMARTS simulator
```bash
conda activate smarts
```

Install [SMARTS simulator](https://github.com/huawei-noah/SMARTS).
```bash
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"
```

## Install Pytorch
```bash
conda install pytorch==1.12.0 -c pytorch
```

## Training
Run the. You can speficy
```bash
python train.py
```

## Testing
Run the testing. You can speficy
```bash
python train.py
```

