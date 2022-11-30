# 2022 NeurIPS Driving SMARTS Competition Track 1 Solution

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
Run `train.py` in the `train` folder. Leave the arguments vacant to use the default setting.
```bash
python train.py --epochs 1000 --batch_size 64 --learning_rate 2e-4 --device cuda
```

### Testing
Run `test.py` in the `submission` folder to test the framework with our pretrained model. You can set `--envision` to visualize the performance of the framework in envision or set `--sumo` to visualize in sumo.
```bash
python test.py --episodes 50 --envisoin
```
To visualize in Envision (some bugs exist in showing the road map), you need to manually start the envision server and then go to `http://localhost:8081/`.
```bash
scl envision start -p 8081
```
