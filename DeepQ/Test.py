import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
import gym_lunar_lander_custom
import argparse
import torch

if __name__ == '__main__':



    env = gym.make('LunarLanderOOD-v0')
    observation = env.reset()
    print("observation")
    print(observation)
    