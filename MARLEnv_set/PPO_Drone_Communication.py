
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
#import gym

from MARL_evns.env_Drones.env_Drones import *

from PIL import Image
import numpy as np

import argparse


from transformer_DL import TransformerEncoder,TransformerEncoderLayer


from quantize import Quantize

from QuantizerFunction import QuantizerFunction

parser = argparse.ArgumentParser()

parser.add_argument('--Round', type=int, default=None,
					help='Round of Random seeds')


parser.add_argument('--data', type=str, default="Drone",
					help='which data to use')

parser.add_argument('--StateCommunication', type=str, default="Cross_attention_SA",
					help='how different agency communicate')


parser.add_argument('--N_agents', type=int, default=3,
					help='Number of Agents')


parser.add_argument('--Method', type=str, default="Original",
					help='Method')

args = parser.parse_args()

###build Env


view_range=10
map_size=50
N_agents=args.N_agents
#train_env = EnvDrones(map_size, N_agents, view_range, 50, 50)   # map_size, drone_num, view_range, tree_num, human_num
#train_env.rand_reset_drone_pos()

train_env1 = EnvDrones(map_size, N_agents, view_range, 20, 100)   # map_size, drone_num, view_range, tree_num, human_num
train_env1.rand_reset_drone_pos()


train_env2 = EnvDrones(map_size, N_agents, view_range, 100, 20)   # map_size, drone_num, view_range, tree_num, human_num
train_env2.rand_reset_drone_pos()


train_env3 = EnvDrones(map_size, N_agents, view_range, 20, 20)   # map_size, drone_num, view_range, tree_num, human_num
train_env3.rand_reset_drone_pos()


train_env4 = EnvDrones(map_size, N_agents, view_range, 100, 100)   # map_size, drone_num, view_range, tree_num, human_num
train_env4.rand_reset_drone_pos()
ListENv_train=[train_env1,train_env2,train_env3,train_env4]

# train_env1 = EnvDrones(map_size, N_agents, view_range, 50, 100)   # map_size, drone_num, view_range, tree_num, human_num
# train_env1.rand_reset_drone_pos()

test_env1 = EnvDrones(map_size, N_agents, view_range, 20, 100)   # map_size, drone_num, view_range, tree_num, human_num
test_env1.rand_reset_drone_pos()


test_env2 = EnvDrones(map_size, N_agents, view_range, 100, 20)   # map_size, drone_num, view_range, tree_num, human_num
test_env2.rand_reset_drone_pos()


test_env3 = EnvDrones(map_size, N_agents, view_range, 20, 20)   # map_size, drone_num, view_range, tree_num, human_num
test_env3.rand_reset_drone_pos()


test_env4 = EnvDrones(map_size, N_agents, view_range, 100, 100)   # map_size, drone_num, view_range, tree_num, human_num
test_env4.rand_reset_drone_pos()
ListENv_test=[test_env1,test_env2,test_env3,test_env4]




test_env = EnvDrones(map_size, N_agents, view_range, 150, 5)   # map_size, drone_num, view_range, tree_num, human_num
test_env.rand_reset_drone_pos()



OODtest_env=EnvDrones(map_size, N_agents, view_range, 5, 50)   # map_size, drone_num, view_range, tree_num, human_num
OODtest_env.rand_reset_drone_pos()

EpisodeLength=25

SEED = args.Round

#train_env.seed(SEED);
#test_env.seed(SEED+1);
np.random.seed(SEED);
torch.manual_seed(SEED);



class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
		super().__init__()

		self.fc_1 = nn.Linear(input_dim, hidden_dim)
		self.fc_2 = nn.Linear(hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		x = self.fc_1(x)
		x = self.dropout(x)
		x = F.relu(x)
		x = self.fc_2(x)
		return x


class ActorCritic_attention(nn.Module):
	def __init__(self,args, INPUT_DIM=1183, HIDDEN_DIM=128, OUTPUT_DIM=4,N_rules=3,key_dim=128):
		super().__init__()

		self.key_dim=key_dim
		self.N_rules=N_rules

		self.args=args

		# self.actors=nn.ModuleList()
		# self.critics=nn.ModuleList()
		# for i in range(N_rules):
		# 	actor=MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
		# 	critic=MLP(INPUT_DIM, HIDDEN_DIM, 1) 
		# 	self.actors.append(actor)
		# 	self.critics.append(critic)

		self.actors=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM) for i in range(args.N_agents)])
		self.critics=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, 1) for i in range(args.N_agents)])
		

		self.projector=nn.Linear(INPUT_DIM, HIDDEN_DIM)



		if args.StateCommunication=="Cross_attention_SA":
			self.Cross_object_SelfAttention=torch.nn.MultiheadAttention(embed_dim=INPUT_DIM, num_heads=1)


		elif args.StateCommunication=="Independent":
			pass


		###discretization function 

		self.N_tightness_levels=3
		self.CodebookSize=128

		self.N_factors=[8,4,1]
		self.alpha=0.1###hyperparameter control penalizaing term for using more factors

		#self.QuantizeFunctions=nn.ModuleList([Quantize(HIDDEN_DIM,self.CodebookSize,N_factor) for N_factor in self.N_factors])	
		self.QuantizeFunctions=QuantizerFunction(HIDDEN_DIM,48,self.args)
		###keys for the quantization modules 

		self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,HIDDEN_DIM))

		self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=4)


	def forward(self, state):


		####inter-agent communication

		if self.args.StateCommunication=="Cross_attention_SA":

			message,_=self.Cross_object_SelfAttention(state,state,state)

		if self.args.StateCommunication=="Independent":
			message=torch.zeros(state.shape)

		






		####combine state and message 
		state=state+message
		
		state=self.projector(state)


		####quantizateion function 
		state,ExtraLoss,att_scores=self.QuantizeFunctions(state)

		


		actions_pred, value_pred=[],[]
		for i in range(args.N_agents):
			action_pred_vec = self.actors[i](state[i,:,:])
			value_pred_vec = self.critics[i](state[i,:,:])

			actions_pred.append(action_pred_vec.unsqueeze(1))
			value_pred.append(value_pred_vec.unsqueeze(1))

		actions_pred=torch.cat(actions_pred,1).permute(1,0,2)

		value_pred=torch.cat(value_pred,1).permute(1,0,2)


		return actions_pred,value_pred,ExtraLoss,att_scores.sum(1).squeeze(0).detach().clone()


#INPUT_DIM = train_env.map_size**2
HIDDEN_DIM = 128
OUTPUT_DIM = 4

policy = ActorCritic_attention(args=args)

# if args.RuleSelection=="Attention":
# 	policy = ActorCritic_attention(args=args,N_rules=args.N_Rules)
# elif args.RuleSelection=="Shared":
# 	policy = ActorCritic_shared(args=args)
# elif args.RuleSelection=="Separate":
# 	policy =ActorCritic_separate(N_agents=N_agents)

# for name, param in policy.named_parameters():
#     if param.requires_grad:
#         print(name)
#         print(param.data)

model_parameters = filter(lambda p: p.requires_grad, policy.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of policy parameters")
print(params)
####project for GWS

LEARNING_RATE = 0.01

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)


def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0)

policy.apply(init_weights)





def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
		
	policy.train()
		
	states = []
	actions = []
	log_prob_actions = []
	values = []
	rewards = []
	done = False
	episode_reward = 0

	#state = env.reset()
	env.rand_reset_drone_pos()
	#env.reset_drone_pos()

	joint_obs=env.get_joint_obs()
	joint_obs=(joint_obs*255).astype(np.uint8)
	im = Image.fromarray(joint_obs)
	im.save("joint_obs_"+"Initial"+".png")


	All_ExtraLoss=0
	All_attScores=[]
	for StepN in range(EpisodeLength):

		state=[]
		for indx in range(len(env.drone_list)):
			Obs_drone=env.get_drone_obs(env.drone_list[indx])
			Obs_drone=torch.FloatTensor(Obs_drone).unsqueeze(0)
			Obs_drone=Obs_drone.flatten().unsqueeze(0)


			# print("Obs_drone")
			# print(Obs_drone.shape)


			DronePosition=env.drone_list[indx].pos
			# print("DronePosition")
			# print(DronePosition)
			X_position=torch.zeros(1,map_size)
			X_position[:,DronePosition[0]]=1
			
			Y_position=torch.zeros(1,map_size)
			Y_position[:,DronePosition[1]]=1
			
			allInfor=torch.cat([Obs_drone,X_position,Y_position],1)
			# print("allInfor")
			# print(allInfor.shape)
			state.append(allInfor)

		state=torch.cat(state,0).unsqueeze(1)#(T,1,state_dim)


		######self-attention on states
				

		#append state here, not after we get the next state from env.step()
		action_pred,value_pred,ExtraLoss, attscores = policy(state)

		All_ExtraLoss+=float(ExtraLoss)
		All_attScores.append(attscores.unsqueeze(0))


		value_pred=value_pred.unsqueeze(1).unsqueeze(1)

		states.append(state.detach().clone())
		
	
		Temperature=1
		action_prob = F.softmax(action_pred/Temperature, dim = -1)


		dist = distributions.Categorical(action_prob)



		action = dist.sample()


		log_prob_action = dist.log_prob(action)




		########movement of human, drone 

		human_act_list = []
		for i in range(env.human_num):
			####human move randomly
			human_act_list.append(random.randint(0, 3))

		# print("human_act_list")
		# print(human_act_list)

		drone_act_list=action.flatten().tolist()
		# print("drone action")
		# print(drone_act_list)	


		env.step(human_act_list, drone_act_list)





		###reward

		joint_obs=env.get_joint_obs()


		reward_droneobs=[]
		for indx in range(len(env.drone_list)):
			Obs_drone=env.get_drone_obs(env.drone_list[indx])
			Obs_drone=(Obs_drone*255).astype(np.uint8)
			#im = Image.fromarray(Obs_drone)
			#im.save("Obs_drone_"+str(indx)+"_"+str(StepN)+".png")

			reward_droneobs_single=torch.tensor(10*float(np.sum((Obs_drone[:,:,0]==1)*(Obs_drone[:,:,1]==0)*(Obs_drone[:,:,2]==0))))
			reward_droneobs.append(reward_droneobs_single)			

		reward_droneobs=torch.tensor(reward_droneobs).unsqueeze(1).unsqueeze(1)
		# print("reward_droneobs")
		# print(reward_droneobs.shape)

		##now negative
		reward_jointview=-torch.tensor(10*float(np.sum((joint_obs[:,:,0]==1)*(joint_obs[:,:,1]==0)*(joint_obs[:,:,2]==0)))).repeat(value_pred.shape[0]).unsqueeze(1).unsqueeze(1)
		reward_stepcost=torch.tensor(-1+float(torch.randn(1))/10000).repeat(value_pred.shape[0]).unsqueeze(1).unsqueeze(1)
		

		reward=reward_jointview+reward_stepcost



		joint_obs=(joint_obs*255).astype(np.uint8)
		im = Image.fromarray(joint_obs)
		im.save("joint_obs_"+str(StepN)+".png")


		# full_obs=env.get_full_obs()
		# # print("full view")
		# # print(np.sum((full_obs[:,:,0]==1)*(full_obs[:,:,1]==0)*(full_obs[:,:,2]==0)))

		# full_obs=(full_obs*255).astype(np.uint8)

		# im = Image.fromarray(full_obs)


		actions.append(action)
		log_prob_actions.append(log_prob_action)
		values.append(value_pred)
		rewards.append(reward)
		
		episode_reward += reward
	
	states = torch.cat(states,1)
	actions = torch.cat(actions,1)    
	log_prob_actions = torch.cat(log_prob_actions,1)
	values = torch.cat(values,1).squeeze(2)
	rewards=torch.cat(rewards,1).squeeze(2)


	returns=[]
	for agent_idx in range(rewards.shape[0]):	
		return_vec = calculate_returns(rewards[agent_idx,:].flatten().tolist(), discount_factor)
		return_vec =return_vec.unsqueeze(0)

		returns.append(return_vec)
	returns=torch.cat(returns,0)


	advantages = calculate_advantages(returns, values)


	policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip,args)

	# for name, param in policy.named_parameters():
	#     if param.requires_grad and name=="Cross_object_SelfAttention.out_proj.bias":
	#         print(name)
	#         print(param.data)

	All_attScores=torch.cat(All_attScores,0)

	return policy_loss, value_loss, episode_reward,All_ExtraLoss,All_attScores



def calculate_returns(rewards, discount_factor, normalize = True):
	
	returns = []
	R = 0
	
	for r in reversed(rewards):
		R = r + R * discount_factor
		returns.insert(0, R)
		
	returns = torch.tensor(returns)
	
	if normalize:
		returns = (returns - returns.mean()) / returns.std()
		
	return returns


def calculate_advantages(returns, values, normalize = True):
	

	advantages = returns - values
	
	if normalize:
		
		advantages = (advantages - advantages.mean()) / advantages.std()
		
	return advantages


def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip,args):
	
	total_policy_loss = 0 
	total_value_loss = 0
	
	advantages = advantages.detach()
	log_prob_actions = log_prob_actions.detach()
	actions = actions.detach()
	
	for _ in range(ppo_steps):
				
		#get new log prob of actions for all input states
		action_pred, value_pred,ExtraLoss,attscores= policy(states)

		
		value_pred = value_pred.squeeze(-1)

		Temperature=1
		action_prob = F.softmax(action_pred/Temperature, dim = -1)


		#KLLoss=(action_prob_internal*(action_prob_internal.log()-action_prob.log())).sum()

		dist = distributions.Categorical(action_prob)
		
		#new log prob using old actions

		new_log_prob_actions = dist.log_prob(actions.reshape((actions.shape[0],actions.shape[1])))
		

		policy_ratio = (new_log_prob_actions - log_prob_actions).exp()	
		policy_loss_1 = policy_ratio * advantages
		policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
		
		policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()


		beta=0.1
		policy_loss=policy_loss+beta*ExtraLoss


		value_loss = F.smooth_l1_loss(returns, value_pred.reshape(returns.shape)).sum()
	
		optimizer.zero_grad()

		
			
		policy_loss.backward(retain_graph=True)
		value_loss.backward(retain_graph=True)




		optimizer.step()
	
		total_policy_loss += policy_loss.item()
		total_value_loss += value_loss.item()
	
	return total_policy_loss / ppo_steps, total_value_loss / ppo_steps




def evaluate(env, policy):
	
	policy.eval()
	
	rewards = []
	done = False
	episode_reward = 0

	#state = env.reset()
	All_attScores=[]
	for i in range(EpisodeLength):

		state=[]
		for indx in range(len(env.drone_list)):
			Obs_drone=env.get_drone_obs(env.drone_list[indx])
			Obs_drone=torch.FloatTensor(Obs_drone).unsqueeze(0)
			Obs_drone=Obs_drone.flatten().unsqueeze(0)


			# print("Obs_drone")
			# print(Obs_drone.shape)


			DronePosition=env.drone_list[indx].pos
			# print("DronePosition")
			# print(DronePosition)
			X_position=torch.zeros(1,map_size)
			X_position[:,DronePosition[0]]=1
			
			Y_position=torch.zeros(1,map_size)
			Y_position[:,DronePosition[1]]=1
			
			allInfor=torch.cat([Obs_drone,X_position,Y_position],1)
			# print("allInfor")
			# print(allInfor.shape)
			state.append(allInfor)

		state=torch.cat(state,0).unsqueeze(1)#(T,1,state_dim)


	
		#append state here, not after we get the next state from env.step()
		
		with torch.no_grad():
			action_pred,value_pred,ExtraLoss,attscores= policy(state)
			All_attScores.append(attscores.unsqueeze(0))


			value_pred=value_pred.unsqueeze(1).unsqueeze(1)

			
		
			Temperature=1
			action_prob = F.softmax(action_pred/Temperature, dim = -1)

			
			
			dist = distributions.Categorical(action_prob)



			action = dist.sample()


			log_prob_action = dist.log_prob(action)




		########movement of human, drone 

		human_act_list = []
		for i in range(env.human_num):
			####human move randomly
			human_act_list.append(random.randint(0, 3))

		# print("human_act_list")
		# print(human_act_list)

		drone_act_list=action.flatten().tolist()
		# print("drone action")
		# print(drone_act_list)	


		env.step(human_act_list, drone_act_list)





		###reward

		joint_obs=env.get_joint_obs()


		reward_droneobs=[]
		for indx in range(len(env.drone_list)):
			Obs_drone=env.get_drone_obs(env.drone_list[indx])
			Obs_drone=(Obs_drone*255).astype(np.uint8)
			#im = Image.fromarray(Obs_drone)
			#im.save("Obs_drone_"+str(indx)+"_"+str(StepN)+".png")

			reward_droneobs_single=torch.tensor(10*float(np.sum((Obs_drone[:,:,0]==1)*(Obs_drone[:,:,1]==0)*(Obs_drone[:,:,2]==0))))
			reward_droneobs.append(reward_droneobs_single)			

		reward_droneobs=torch.tensor(reward_droneobs).unsqueeze(1).unsqueeze(1)
		# print("reward_droneobs")
		# print(reward_droneobs.shape)

		##now negative
		reward_jointview=-torch.tensor(10*float(np.sum((joint_obs[:,:,0]==1)*(joint_obs[:,:,1]==0)*(joint_obs[:,:,2]==0)))).repeat(value_pred.shape[0]).unsqueeze(1).unsqueeze(1)
		reward_stepcost=torch.tensor(-1+float(torch.randn(1))/10000).repeat(value_pred.shape[0]).unsqueeze(1).unsqueeze(1)
		

		reward=reward_jointview+reward_stepcost



		###move 
		env.step(human_act_list, drone_act_list)

		episode_reward += reward



	All_attScores=torch.cat(All_attScores,0)
		
	return episode_reward,All_attScores




MAX_EPISODES = 50
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 999999
PRINT_EVERY = 1
PPO_STEPS = 5
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []
OODtest_rewards = []

train_mean_rewards=[]
test_mean_rewards=[]
OODtest_mean_rewards=[]



for episode in range(1, MAX_EPISODES+1):

	import random
	inx=random.choice(list(range(len(ListENv_train))))
	print("env chosen",str(inx))
	train_env=ListENv_train[inx]

	test_env=ListENv_test[inx]
	
	policy_loss, value_loss, train_reward,ExtraLoss_train,AllAttScores_train= train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)

	test_reward,AllAttScores_test = evaluate(test_env, policy)

	OODtest_reward,AllAttScores_OODtest = evaluate(OODtest_env,policy)
	
	train_rewards.append(train_reward)
	test_rewards.append(test_reward)
	OODtest_rewards.append(OODtest_reward)
	
	

	
	mean_train_rewards = torch.cat(train_rewards,1)[:,-N_TRIALS:,].mean().item()
	mean_test_rewards = torch.cat(test_rewards,1)[:,-N_TRIALS:,].mean().item()
	OODmean_test_rewards = torch.cat(OODtest_rewards,1)[:,-N_TRIALS:,].mean().item()
	

	train_mean_rewards.append(mean_train_rewards)
	test_mean_rewards.append(mean_test_rewards)
	OODtest_mean_rewards.append(OODmean_test_rewards)
	
	
	import csv   
	fields=[args.data,args.N_agents,args.Method,episode,mean_train_rewards,mean_test_rewards,OODmean_test_rewards,ExtraLoss_train,AllAttScores_train.sum(0).tolist(),AllAttScores_test.sum(0).tolist(),AllAttScores_OODtest.sum(0).tolist()]
	print("fileds")
	print(fields)
	with open(r'../../MARL_Results.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow(fields)



	if episode % PRINT_EVERY == 0:
	
		print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Mean OOD Test Rewards: {OODmean_test_rewards:5.1f} |')
	
	if mean_test_rewards >= REWARD_THRESHOLD:
		
		print(f'Reached reward threshold in {episode} episodes')
		
		break


# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# Data=pd.DataFrame({"Episode":[i for i in range(len(train_mean_rewards))],
# 					"train_mean_rewards":train_mean_rewards,
# 					"test_mean_rewards":train_mean_rewards})


# fig=plt.figure()

# plt.plot(["Episode","Episode"], ["train_mean_rewards","test_mean_rewards"], data=Data)

# plt.title("Results")
# plt.xlabel("Episode")
# plt.ylabel("mean reward")
# plt.savefig("Results.png")