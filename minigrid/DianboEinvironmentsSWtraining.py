from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
#A new set of enviroments designed by Dianbo to training and test OOD generalization


###################environemnt with specific parameters for schema weight mapper training##########

class DLEnv_SW_training(MiniGridEnv):
	"""
	Environment in which the agent has to fetch a random object
	named using English text strings
	Environment with a random room size 
	with random number of objects with random color 
	"""

	def __init__(
		self,
		width=5,
		height=9,
		Max_numObjs=1,
		Max_size=9,
		Min_size=5,
		NothingOrWallOrLava=0,
		ObjectTypes=["box"],
		FinalGoal="Objects",
		agent_start_pos=None,
		agent_start_dir=None
	):
		
		#self.Min_size=Min_size
		self.Max_size=Max_size
		self.Min_size=Min_size

		self.Max_numObjs=Max_numObjs

		self.width=width
		self.height=height
		self.NothingOrWallOrLava=NothingOrWallOrLava
		self.FinalGoal=FinalGoal
		self.ObjectTypes=ObjectTypes
		
		self.agent_start_pos=agent_start_pos
		self.agent_start_dir=agent_start_dir

		Name="DLSWEnvW"+str(width)+"H"+str(height)+"T"+str(NothingOrWallOrLava)+FinalGoal
		DLEnv_SW_training.__name__=Name #assign name to different subclasses



		super().__init__(
			#this maximum size depend on setting
			grid_size=Max_size,
			max_steps=5*Max_size**2,
			# Set this to True for maximum speed
			see_through_walls=True
		)

	def _gen_grid(self, width, height):
		#in each episode the size of the room varies 
		#size=random.choice(range(self.Min_size,self.Max_size+1))
		width=self.width
		height=self.height

		self.numObjs=random.choice(range(1,self.Max_numObjs+1))

		# Create an empty grid
		self.grid = Grid(width, height)
		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)
		#decide if we need a wall-door-key or lava rows
		NothingOrWallOrLava=self.NothingOrWallOrLava

		if NothingOrWallOrLava==0:
			# Nothing
			# Randomize the player start position and orientation if not given
			if self.agent_start_pos is not None:
				self.agent_pos = self.agent_start_pos
				self.agent_dir = self.agent_start_dir
			else:
				self.place_agent()

		if NothingOrWallOrLava==1:
			# Create a vertical splitting wall a door and a key 
			 # Create a vertical splitting wall
			splitIdx = self._rand_int(2, width-2)
			self.grid.vert_wall(splitIdx, 0)

			# Place the agent at a random position and orientation
			# on the left side of the splitting wall
			self.place_agent(size=(splitIdx, height))

			# Place a door in the wall
			doorIdx = self._rand_int(1, height-2)
			self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

			# Place a yellow key on the left side
			self.place_obj(
			    obj=Key('yellow'),
			    top=(0, 0),
			    size=(splitIdx, height)
			)


	    
		if NothingOrWallOrLava==2:
			# Randomly place the lava rows or nothings		
			self.strip2_row=random.choice([-1,2,3])
			if self.strip2_row>0:
				for i in range(self.width - self.Min_size+1):
					self.grid.set(self.Min_size-2+i, 1, Lava())
					self.grid.set(self.Min_size-2+i, self.strip2_row, Lava())
			# Randomize the player start position and orientation if not given
			if self.agent_start_pos is not None:
				self.agent_pos = self.agent_start_pos
				self.agent_dir = self.agent_start_dir
			else:
				self.place_agent()
		
		#choose whether the agent is suposed to a square or pick up an objects 


		if self.FinalGoal=="Objects":
			# Choose a random object to be picked up
			#generate and place object 

			types = self.ObjectTypes#only box is used during training, ball will be used for testing

			objs = []

			# For each object to be generated
			while len(objs) < self.numObjs:
				objType = self._rand_elem(types)
				objColor = self._rand_elem(COLOR_NAMES)
				#objColor = 'green'#use only green color to simplify envs.

				#if objType == 'key':
				#	obj = Key(objColor)
				#if objType == 'ball':
				#	obj = Ball(objColor)
				if objType == 'box':
					obj = Box(objColor)

				self.place_obj(obj)
				objs.append(obj)
			#set the final target
			target = objs[self._rand_int(0, len(objs))]
			self.targetType = target.type
			self.targetColor = target.color

			descStr = '%s %s' % (self.targetColor, self.targetType)

			# Generate the mission string ( this string is not used in the model training)
			idx = self._rand_int(0, 5)
			if idx == 0:
				self.mission = 'get a %s' % descStr
			elif idx == 1:
				self.mission = 'go get a %s' % descStr
			elif idx == 2:
				self.mission = 'fetch a %s' % descStr
			elif idx == 3:
				self.mission = 'go fetch a %s' % descStr
			elif idx == 4:
				self.mission = 'you must fetch a %s' % descStr
			assert hasattr(self, 'mission')

		if self.FinalGoal == "Goal":
			# Place a goal square randomly
			self.put_obj(Goal(),width - 2, height - 2)
			self.mission = "get to the green goal square"

	def step(self, action):
		obs, reward, done, info = MiniGridEnv.step(self, action)

		if self.FinalGoal=="Objects":
			if self.carrying:
				if self.carrying.color == self.targetColor and \
				   self.carrying.type == self.targetType:
					reward = self._reward()
					done = True
				else:
					reward = 0
					done = True

		if self.FinalGoal=="Goal":
				#additonal codes not needed
				1+1

							
		return obs, reward, done, info
#############register  the environments##############
import argparse
import time
import numpy
import torch

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--Width", type=int,required=True)
parser.add_argument("--Height", type=int,required=True)
parser.add_argument("--FinalGoal", type=str,required=True)
parser.add_argument("--NothingOrWallOrLava", type=int,required=True)

args = parser.parse_args()
Width=args.Width
Height=args.Height
FinalGoal=args.FinalGoal
NothingOrWallOrLava=args.NothingOrWallOrLava


class DLSWTraining(DLEnv_SW_training):
	def __init__(self, **kwargs):
		super().__init__(width=Width,
		height=Height,
		NothingOrWallOrLava=NothingOrWallOrLava,
		FinalGoal=FinalGoal,**kwargs)

Name="DLSWEnvW"+str(Width)+"H"+str(Height)+"T"+str(NothingOrWallOrLava)+FinalGoal
print("Env name:"+'MiniGrid-'+Name+'-random-v0')
register(
	id='MiniGrid-'+Name+'-random-v0',
	entry_point='gym_minigrid.envs:DLSWTraining'
)