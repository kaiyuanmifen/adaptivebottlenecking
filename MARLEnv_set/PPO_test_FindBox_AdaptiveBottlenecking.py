
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

from MARL_evns.env_FindBox.env_FindBox import *

from PIL import Image
import numpy as np

import argparse


from transformer_DL import TransformerEncoder,TransformerEncoderLayer



from quantize import Quantize



parser = argparse.ArgumentParser()

parser.add_argument('--Round', type=int, default=None,
					help='Round of Random seeds')

parser.add_argument('--Method', type=str, default="Original",
					help='which method to use')

parser.add_argument('--data', type=str, default="Drone",
					help='which data to use')


parser.add_argument('--N_agents', type=int, default=6,
					help='Number of Agents')

args = parser.parse_args()

###build Env


map_size=15
N_agents=args.N_agents
train_env = EnvFindBox(map_size)  # map_size, drone_num, view_range, tree_num, human_num
train_env.reset()


test_env = EnvFindBox(map_size)   # map_size, drone_num, view_range, tree_num, human_num
test_env.reset()


OODtest_env=EnvFindBox(30)   # map_size, drone_num, view_range, tree_num, human_num
OODtest_env.reset()

EpisodeLength=300

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
	def __init__(self,args, INPUT_DIM=3*3*3+2, HIDDEN_DIM=128, OUTPUT_DIM=4,key_dim=128):
		super().__init__()

		self.key_dim=key_dim

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


		###discretization function 

		self.N_tightness_levels=3
		self.CodebookSize=16

		self.QuantizeFunctions=nn.ModuleList([Quantize(HIDDEN_DIM,self.CodebookSize,int(4//(2**j))) for j in range(self.N_tightness_levels)])
		
		###keys for the quantization modules 

		self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,HIDDEN_DIM))

		self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=4)

	def forward(self, state):

		state=self.projector(state)

		bsz,T,Hsz=state.shape

		if self.args.Method=="Quantization":
			state=state.reshape(bsz*T,1,Hsz)
			state,CBloss,ind=self.QuantizeFunctions[1](state)#use a fixed function to discretize

			state=state.reshape(bsz,T,Hsz)


		if self.args.Method=="Adaptive_Quantization":
			
			###key-query attention to decide which quantization_function to use
			query=state.reshape(bsz*T,1,Hsz)
			
			_,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

			att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2)
	
			state=state.reshape(bsz*T,1,Hsz)

			Zs=[]
			CBloss=0
			for i in range(self.N_tightness_levels):
				Z,CBloss_vec,_=self.QuantizeFunctions[i](state)#use a fixed function to discretize
				Zs.append(Z)
				CBloss+=CBloss_vec#sum up the codebookloss


			CBloss=CBloss/self.N_tightness_levels
			Zs=torch.cat(Zs,1)
	
			state=torch.bmm(att_scores.permute(1,0,2),Zs)

			state=state.reshape(bsz,T,Hsz)

		elif self.args.Method=="Original":

			CBloss=0




		actions_pred, value_pred=[],[]
		for i in range(args.N_agents):
			action_pred_vec = self.actors[i](state[i,:,:])
			value_pred_vec = self.critics[i](state[i,:,:])

			actions_pred.append(action_pred_vec.unsqueeze(1))
			value_pred.append(value_pred_vec.unsqueeze(1))

		actions_pred=torch.cat(actions_pred,1).permute(1,0,2)

		value_pred=torch.cat(value_pred,1).permute(1,0,2)


		return actions_pred,value_pred,CBloss



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

LEARNING_RATE = 0.001

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
	episode_reward = 0.0

	#state = env.reset()
	env.reset()
	#env.reset_drone_pos()

	joint_obs=env.get_global_obs()
	joint_obs=(joint_obs*255).astype(np.uint8)
	im = Image.fromarray(joint_obs)
	im.save("Images/"+args.data+"joint_obs_"+"Initial"+".png")

	State_memory=[]#record states from each time step

	All_CBLoss=0

	for StepN in range(EpisodeLength):

		state=[]
		for indx in range(args.N_agents):
			if indx==0:
				Obs_agent=env.get_agt1_obs()
				Obs_agent=torch.FloatTensor(Obs_agent).unsqueeze(0)
				Obs_agent=Obs_agent.flatten().unsqueeze(0)


			elif indx==1:
				Obs_agent=env.get_agt2_obs()
				Obs_agent=torch.FloatTensor(Obs_agent).unsqueeze(0)
				Obs_agent=Obs_agent.flatten().unsqueeze(0)

			AgentPosition=torch.tensor(env.get_state().reshape(args.N_agents,2))

			allInfor=torch.cat([Obs_agent,AgentPosition[indx,:].unsqueeze(0)],1).float()
		
			state.append(allInfor)

		state=torch.cat(state,0).unsqueeze(1)#(N_Agent,1,state_dim)
		State_memory.append(state.unsqueeze(0))
		######self-attention on states
				

		#append state here, not after we get the next state from env.step()
		action_pred,value_pred,CBloss = policy(state)


	

		All_CBLoss+=CBloss
		value_pred=value_pred.unsqueeze(1).unsqueeze(1)

		states.append(state.detach().clone())
		
	
		Temperature=1
		action_prob = F.softmax(action_pred/Temperature, dim = -1)
		# print("action_prob")
		# print(action_prob)
		
		dist = distributions.Categorical(action_prob)



		action = dist.sample()
		

		log_prob_action = dist.log_prob(action)



		reward,done=env.step(action.flatten().tolist())
		reward=torch.tensor(reward).reshape(1,1,1).repeat(args.N_agents,1,1)
		


		# import random

		# if random.random()<0.001:
		# 	joint_obs=env.get_global_obs()
		# 	joint_obs=(joint_obs*255).astype(np.uint8)
		# 	im = Image.fromarray(joint_obs)
		# 	im.save("Images/"+args.data+"joint_obs_"+str(StepN)+".png")


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
		
		if done:
			print("Done")
			break

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


	policy_loss, value_loss = update_policy(policy, states,State_memory, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

	return policy_loss, value_loss, episode_reward,All_CBLoss



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


def update_policy(policy, states,State_memory, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
	
	total_policy_loss = 0 
	total_value_loss = 0
	
	advantages = advantages.detach()
	log_prob_actions = log_prob_actions.detach()
	actions = actions.detach()
	
	for _ in range(ppo_steps):
				
		#get new log prob of actions for all input states
		action_pred, value_pred, CBloss = policy(states)

		
		value_pred = value_pred.squeeze(-1)

		action_prob = F.softmax(action_pred, dim = -1)


		dist = distributions.Categorical(action_prob)
		
		#new log prob using old actions

		new_log_prob_actions = dist.log_prob(actions.reshape((actions.shape[0],actions.shape[1])))
		

		policy_ratio = (new_log_prob_actions - log_prob_actions).exp()	
		policy_loss_1 = policy_ratio * advantages
		policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
		
		policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
		


		value_loss = F.smooth_l1_loss(returns, value_pred.reshape(returns.shape)).sum()
	
		optimizer.zero_grad()


		policy_loss=policy_loss+CBloss
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
	episode_reward = 0.0

	#state = env.reset()
	State_memory=[]#record states from each time step
	for StepN in range(EpisodeLength):

		state=[]
		for indx in range(args.N_agents):
			if indx==0:
				Obs_agent=env.get_agt1_obs()
				Obs_agent=torch.FloatTensor(Obs_agent).unsqueeze(0)
				Obs_agent=Obs_agent.flatten().unsqueeze(0)


			elif indx==1:
				Obs_agent=env.get_agt2_obs()
				Obs_agent=torch.FloatTensor(Obs_agent).unsqueeze(0)
				Obs_agent=Obs_agent.flatten().unsqueeze(0)

			AgentPosition=torch.tensor(env.get_state().reshape(args.N_agents,2))

			allInfor=torch.cat([Obs_agent,AgentPosition[indx,:].unsqueeze(0)],1).float()
		
			state.append(allInfor)


		state=torch.cat(state,0).unsqueeze(1)#(N_Agent,1,state_dim)
		State_memory.append(state.unsqueeze(0))
		######self-attention on states
				
		with torch.no_grad():
		#append state here, not after we get the next state from env.step()
			action_pred,value_pred,CBloss = policy(state)


			value_pred=value_pred.unsqueeze(1).unsqueeze(1)

			
		
			Temperature=1
			action_prob = F.softmax(action_pred/Temperature, dim = -1)
			# print("action_prob")
			# print(action_prob)
			
			dist = distributions.Categorical(action_prob)



			action = dist.sample()
			

			log_prob_action = dist.log_prob(action)




			reward,done=env.step(action.flatten().tolist())
			reward=torch.tensor(reward).reshape(1,1,1).repeat(args.N_agents,1,1)


			# full_obs=env.get_full_obs()
			# # print("full view")
			# # print(np.sum((full_obs[:,:,0]==1)*(full_obs[:,:,1]==0)*(full_obs[:,:,2]==0)))

			# full_obs=(full_obs*255).astype(np.uint8)

			# im = Image.fromarray(full_obs)

			episode_reward += reward
			
			if done:
				break

	
	
	return episode_reward




MAX_EPISODES = 1000
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
	
	policy_loss, value_loss, train_reward, CBloss = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)

	test_reward = evaluate(test_env, policy)

	OODtest_reward = evaluate(OODtest_env,policy)
	
	train_rewards.append(train_reward.mean().item())
	test_rewards.append(test_reward.mean().item())
	OODtest_rewards.append(OODtest_reward.mean().item())
	
	

	
	# mean_train_rewards = torch.cat(train_rewards,1)[:,-N_TRIALS:,].mean().item()
	# mean_test_rewards = torch.cat(test_rewards,1)[:,-N_TRIALS:,].mean().item()
	# OODmean_test_rewards = torch.cat(OODtest_rewards,1)[:,-N_TRIALS:,].mean().item()
	

	# train_mean_rewards.append(mean_train_rewards)
	# test_mean_rewards.append(mean_test_rewards)
	# OODtest_mean_rewards.append(OODmean_test_rewards)
	
	
	import csv   
	fields=[args.data,args.N_agents,args.Method,episode,train_reward.mean().item(),test_reward.mean().item(),OODtest_reward.mean().item(),CBloss]
	print("fileds")
	print(fields)
	with open(r'Results.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow(fields)



	if episode % PRINT_EVERY == 0:
	
		print(f'| Episode: {episode:3} | Mean Train Rewards: {train_reward.mean().item():5.1f} | Mean Test Rewards: {test_reward.mean().item():5.1f} | mean OOD Test Rewards: {OODtest_reward.mean().item():5.1f} |CBloss: {CBloss :5.1f} |')
	
	# if mean_test_rewards >= REWARD_THRESHOLD:
		
	# 	print(f'Reached reward threshold in {episode} episodes')
		
	# 	break


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