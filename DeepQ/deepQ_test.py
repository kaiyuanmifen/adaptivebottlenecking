import gym
from gym import envs

#print(envs.registry.all())
env = gym.make('BipedalWalker-v3')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())