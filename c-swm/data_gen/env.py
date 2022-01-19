"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
	sys.path.insert(0, str(Path.cwd()))

import argparse

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger

#import gym_minigrid

import numpy as np
from PIL import Image


class RandomAgent(object):
	"""The world's simplest agent!"""

	def __init__(self, action_space):
		self.action_space = action_space

	def act(self, observation, reward, done):
		del observation, reward, done
		return self.action_space.sample()


def crop_normalize(img, crop_ratio):
	img = img[crop_ratio[0]:crop_ratio[1]]
	img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
	return np.transpose(np.array(img), (2, 0, 1)) / 255


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
						help='Select the environment to run.')
	parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
						help='Save path for replay buffer.')
	parser.add_argument('--num_episodes', type=int, default=1000,
						help='Total number of episodes to simulate.')
	parser.add_argument('--atari', action='store_true', default=False,
						help='Run atari mode (stack multiple frames).')
	parser.add_argument('--seed', type=int, default=1,
						help='Random seed.')
	args = parser.parse_args()

	logger.set_level(logger.INFO)

	print("evs_ID",args.env_id)
	env = gym.make(args.env_id)

	action_space_dim=env.action_space.n
	print("action_space_dim",action_space_dim)

	np.random.seed(args.seed)
	env.action_space.seed(args.seed)
	env.seed(args.seed)

	agent = RandomAgent(env.action_space)

	episode_count = args.num_episodes
	reward = 0
	done = False

	crop = None
	warmstart = None
	if args.env_id == 'PongDeterministic-v4':
		crop = (35, 190)
		warmstart = 58
	elif args.env_id == 'SpaceInvadersDeterministic-v4':
		crop = (30, 200)
		warmstart = 50
	elif "Deterministic" in args.env_id:#all iid settings
		crop = (35, 190)
		warmstart =58 
	elif "Frameskip-v0" in args.env_id:###OOD1
		crop = (35, 190)
		warmstart =158
	else:					###00D2
		crop = (35, 190)
		warmstart =100


	

	if args.atari:
		if "Frameskip-v0" in args.env_id:###OOD1
			env._max_episode_steps = warmstart + 100
		if "Frameskip-v4" in args.env_id:###OOD2
			env._max_episode_steps = warmstart + 50
			
	replay_buffer = []

	for i in range(episode_count):

		replay_buffer.append({
			'obs': [],
			'action': [],
			'next_obs': []
		})

		ob = env.reset()



		

		
		
		
		if args.atari:
			# Burn-in steps
			for _ in range(warmstart):
				action = agent.act(ob, reward, done)
				ob, _, _, _ = env.step(action)
			
			prev_ob = crop_normalize(ob, crop)
			ob, _, _, _ = env.step(0)
			ob = crop_normalize(ob, crop)

			
			while True:
				replay_buffer[i]['obs'].append(
					np.concatenate((ob, prev_ob), axis=0))
				prev_ob = ob

				action = agent.act(ob, reward, done)
				ob, reward, done, _ = env.step(action)
				ob = crop_normalize(ob, crop)
				# print("obs", ob)
				# print(str(len(ob)))
				# print("action",action)

				replay_buffer[i]['action'].append(action)
				replay_buffer[i]['next_obs'].append(
					np.concatenate((ob, prev_ob), axis=0))

				if done:
					break
		else:

			while True:
				replay_buffer[i]['obs'].append(ob[1])


				action = agent.act(ob, reward, done)
				ob, reward, done, _ = env.step(action)

				

				replay_buffer[i]['action'].append(action)
				replay_buffer[i]['next_obs'].append(ob[1])

				if done:
					break

		if i % 10 == 0:
			print("iter "+str(i))

	env.close()

	
	# Save replay buffer to disk.
	# print("ob:",ob)
	# print(type(ob))
	# print(ob[1].shape)
	# from PIL import Image
	# Example=ob[1].reshape(ob[1].shape[1],ob[1].shape[2],ob[1].shape[0])
	# Example=Example
	# print(Example)
	# #import matplotlib
	# #matplotlib.image.imsave("data/"+args.env_id+".png", ob[1])
	# im = Image.fromarray(Example,'RGB')
	# im.save("data/"+args.env_id+".jpeg")##save one example of the observation as images

	#print("replay_buffer",replay_buffer)
	utils.save_list_dict_h5py(replay_buffer, args.fname)

	###save action space information
	import csv  
	ToSave=[args.env_id,args.fname,action_space_dim]
	with open('data/actionspace.csv', 'a') as fd:
		writer = csv.writer(fd)
		writer.writerow(ToSave)
	