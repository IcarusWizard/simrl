# Simrl
Simrl stands for **S**imple **Im**plementations of **R**einforcement **L**earning. 
This repository is still under development. If you find any problem, feel free to open an issue.

## Installation
```
git clone https://github.com/IcarusWizard/simrl.git
cd simrl 
pip install -e .
```

## Usage
You can start training by calling the algorithm, i.e.
```
python -m simrl.algo.<type>.<name> --env <env_name>
```

## Supported Algorithems
- Model-based
    - PETS
    - MBPO
- Model-free
    - TRPO
    - PPO
    - DQN
    - SAC

## Supported Environments
The default installation only supports classical gym environments with flatten states.
You can enable more supports with `pip install -e .[env]`. Currently, `pybullet` is supported.  
We are planning to add support for more diverse environments and POMDP environments with visual inputs. 