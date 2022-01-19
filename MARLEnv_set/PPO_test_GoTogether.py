
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

from MARL_evns.env_GoTogether.env_GoTogether import *

from PIL import Image
import numpy as np

import argparse


from transformer_DL import TransformerEncoder,TransformerEncoderLayer

parser = argparse.ArgumentParser()

parser.add_argument('--Round', type=int, default=None,
					help='Round of Random seeds')

parser.add_argument('--hypothesis', type=int, default=None,
					help='The hypothesis number (1-5)')

parser.add_argument('--data', type=str, default="Drone",
					help='which data to use')


parser.add_argument('--N_agents', type=int, default=6,
					help='Number of Agents')

args = parser.parse_args()

###build Env


map_size=15
N_agents=args.N_agents
train_env = EnvGoTogether(map_size)  # map_size, drone_num, view_range, tree_num, human_num
train_env.reset()


test_env = EnvGoTogether(map_size)   # map_size, drone_num, view_range, tree_num, human_num
test_env.reset()


OODtest_env=EnvGoTogether(map_size)   # map_size, drone_num, view_range, tree_num, human_num
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


class ActorCritic_attentionSchema(nn.Module):
	def __init__(self,args, INPUT_DIM=map_size*map_size*3+2, HIDDEN_DIM=28,GWS_DIM=8, OUTPUT_DIM=4,key_dim=128):
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

		###encoder from observation to representation 

		self.encoder=MLP(INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM)
		
		if args.hypothesis==1:
			self.attention_module=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, GWS_DIM) for i in range(args.N_agents)])
		
		elif args.hypothesis==2:
			self.attention_module=nn.ModuleList([MLP(GWS_DIM, HIDDEN_DIM, GWS_DIM) for i in range(args.N_agents)])
		
			self.awareness_module=nn.ModuleList([MLP(HIDDEN_DIM, 16, GWS_DIM) for i in range(args.N_agents)])
		
		elif args.hypothesis==3:
			self.attention_module=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, GWS_DIM) for i in range(args.N_agents)])

			self.awareness_module=nn.ModuleList([MLP(GWS_DIM, 16, GWS_DIM) for i in range(args.N_agents)])
		
		elif args.hypothesis==4:
			self.attention_module=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, GWS_DIM) for i in range(args.N_agents)])

			self.awareness_module=nn.ModuleList([MLP(HIDDEN_DIM, 16, GWS_DIM) for i in range(args.N_agents)])
		
		elif args.hypothesis==5:
			self.attention_module=nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, GWS_DIM) for i in range(args.N_agents)])

			self.awareness_module=nn.ModuleList([MLP(HIDDEN_DIM, 16, GWS_DIM) for i in range(args.N_agents)])


		self.actors=nn.ModuleList([MLP(GWS_DIM, HIDDEN_DIM, OUTPUT_DIM) for i in range(args.N_agents)])
		self.critics=nn.ModuleList([MLP(GWS_DIM, HIDDEN_DIM, 1) for i in range(args.N_agents)])
		
		# if self.args.hypothesis==5:
		# 	self.rnn_module = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, 1)

	

	def forward(self, state,State_memory):


		State_memory=torch.cat(State_memory,0).squeeze(2)
		

		State_memory=self.encoder(State_memory)
		# print("state")
		# print(state)
		state=self.encoder(state)
		
		####Project observation to states
		actions_pred, value_pred=[],[]

		#####attention and awareness
		
		GWSes=[]
		ExtraLoss=0#this is the loss that is used for extra comparison

		for i in range(args.N_agents):
			if self.args.hypothesis==1:
				GWS=self.attention_module[i](state[i,:,:])
	
				GWSes.append(GWS.unsqueeze(0))

			elif self.args.hypothesis==2:
				Predicted_GWS=self.awareness_module[i](state[i,:,:])
				GWS=self.attention_module[i](Predicted_GWS)
				GWSes.append(GWS.unsqueeze(0))

			elif self.args.hypothesis==3:
				Attention_results=self.attention_module[i](state[i,:,:])
				GWS=self.awareness_module[i](Attention_results)
				GWSes.append(GWS.unsqueeze(0))

			elif self.args.hypothesis==4:
				Predicted_GWS_awareness=self.awareness_module[i](state[i,:,:])
				GWS=self.attention_module[i](state[i,:,:])
				
				L=torch.nn.MSELoss()
				ExtraLoss+=L(Predicted_GWS_awareness,GWS)

				GWSes.append(GWS.unsqueeze(0))

			elif self.args.hypothesis==5:
				Predicted_GWS_awareness=self.awareness_module[i](state[i,:,:])
				Predicted_GWS_attention=self.attention_module[i](state[i,:,:])
				
				GWS=(Predicted_GWS_awareness+Predicted_GWS_attention)/2

				L=torch.nn.MSELoss()
				ExtraLoss+=L(Predicted_GWS_attention,GWS)+L(Predicted_GWS_awareness,GWS)

				GWSes.append(GWS.unsqueeze(0))




		GWSes=torch.cat(GWSes,0)


		for i in range(args.N_agents):
			action_pred_vec = self.actors[i](GWSes[i,:,:])
			value_pred_vec = self.critics[i](GWSes[i,:,:])

			actions_pred.append(action_pred_vec.unsqueeze(1))
			value_pred.append(value_pred_vec.unsqueeze(1))

		actions_pred=torch.cat(actions_pred,1).permute(1,0,2)

		value_pred=torch.cat(value_pred,1).permute(1,0,2)


		return actions_pred,value_pred,ExtraLoss



#INPUT_DIM = train_env.map_size**2
HIDDEN_DIM = 128
OUTPUT_DIM = 4

policy = ActorCritic_attentionSchema(args=args)

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
	for StepN in range(EpisodeLength):

		state=[]
		for indx in range(args.N_agents):
			Obs_agent=env.get_global_obs()
			Obs_agent=torch.FloatTensor(Obs_agent).unsqueeze(0)
			Obs_agent=Obs_agent.flatten().unsqueeze(0)


	
			AgentPosition=torch.tensor(env.get_state().reshape(args.N_agents,2))
	
			
			allInfor=torch.cat([Obs_agent,AgentPosition[indx,:].unsqueeze(0)],1).float()

			state.append(allInfor)

		state=torch.cat(state,0).unsqueeze(1)#(N_Agent,1,state_dim)
		State_memory.append(state.unsqueeze(0))
		######self-attention on states
				

		#append state here, not after we get the next state from env.step()
		action_pred,value_pred,ExtraLoss = policy(state,State_memory)


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
		


		import random

		if random.random()<0.001:
			joint_obs=env.get_global_obs()
			joint_obs=(joint_obs*255).astype(np.uint8)
			im = Image.fromarray(joint_obs)
			im.save("Images/"+args.data+"joint_obs_"+str(StepN)+".png")


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

	return policy_loss, value_loss, episode_reward



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
		action_pred, value_pred, ExtraLoss = policy(states,State_memory)

		
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

		policy_loss.backward(retain_graph=True)
		value_loss.backward(retain_graph=True)

		if ExtraLoss!=0:
			ExtraLoss.backward(retain_graph=True)
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
			Obs_agent=env.get_global_obs()
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
			action_pred,value_pred,ExtraLoss = policy(state,State_memory)


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
			
	
	return episode_reward




MAX_EPISODES = 500
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
	
	policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
	
	test_reward = evaluate(test_env, policy)

	OODtest_reward = evaluate(OODtest_env,policy)
	
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
	fields=[args.data,args.N_agents,args.hypothesis,episode,mean_train_rewards,mean_test_rewards,OODmean_test_rewards]
	print("fileds")
	print(fields)
	with open(r'Results.csv', 'a') as f:
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