
import torch
import sys
import torch.nn.functional as F
from torch import nn

import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from model import ACModel
import os
import re




class SW_net(torch.nn.Module):
	def __init__(self, din, dout,n_hidden):
		super(SW_net, self).__init__()
		self.hidden = torch.nn.Linear(din, n_hidden)   # hidden layer
		self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
		self.predict = torch.nn.Linear(n_hidden, dout)   # output layer

	def forward(self, x):
		x = F.relu(self.hidden(x))      # activation function for hidden layer
		x = F.relu(self.hidden2(x))      # activation function for hidden layer
		Out = self.predict(x)             # linear output
		return Out


def Predict_SW(Width,Height,BlockType,Targets,din=7,dout=16,n_hidden=6,NUM_UNITS=4,num_schemas=4):
	'''
	This function take in evironmenta features and output schema weighting 
	'''

	Blocks=[0,0,0]
	Blocks[BlockType]=1
	
	if Targets=="Goal":
		Targets=[1,0]
	elif Targets=="Objects":
		Targets=[0,1]
	#print(Targets)

	Features=[]
	Features.append(Width/9.0)#normalization
	Features.append(Height/9.0)
	Features=Features+Blocks+Targets
	Features=torch.FloatTensor(Features)
	#print(Features)
	
	#predict schema weight x2h
	PATH="SW_predictor_x2h.pt"
	Model=SW_net(din,dout,n_hidden)
	Model.load_state_dict(torch.load(PATH))
	SW1x2h=Model(Features)

	#predict schema weight h2h
	PATH="SW_predictor_h2h.pt"
	Model=SW_net(din,dout,n_hidden)
	Model.load_state_dict(torch.load(PATH))
	SW1h2h=Model(Features)


	#reshape the schema weighting
	SW1x2h=SW1x2h.reshape(NUM_UNITS,num_schemas)
	SW1h2h=SW1h2h.reshape(NUM_UNITS,num_schemas)

	return SW1x2h,SW1h2h
	


if __name__=="__main__":
	print("loading data")

	# Parse arguments

	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="MiniGrid-DLEnv-random-v0",
						help="name of the environment (not important for this code)")
	parser.add_argument("--model", default="PlaceHolder",
						help="name of the trained model ")
	parser.add_argument("--episodes", type=int, default=100,
						help="number of episodes of evaluation (default: 100)")
	parser.add_argument("--seed", type=int, default=0,
						help="random seed (default: 0)")
	parser.add_argument("--procs", type=int, default=16,
						help="number of processes (default: 16)")
	parser.add_argument("--argmax", action="store_true", default=False,
						help="action with highest probability is selected")
	parser.add_argument("--worst-episodes-to-show", type=int, default=10,
						help="how many worst episodes to show")
	parser.add_argument("--recurrence", type=int, default=32,
						help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
	parser.add_argument("--text", action="store_true", default=False,
						help="add a GRU to the model to handle text input")

	## Model Parameters
	parser.add_argument('--use_rim',action = 'store_true', default = True,help="whether use RIM")

	parser.add_argument('--rim_type',type=str,default="Shared",
						help="type of RIM to use")

	parser.add_argument('--NUM_UNITS',type=int, default=4,help="Number of RNN units used in the model")

	parser.add_argument('--k',type=int, default=2,help="Number of RNN units active in each time step")

	parser.add_argument('--rnn_cell',type=str, default="LSTM",help="basic type of RNN units, LSTM or GRU")

	parser.add_argument('--UnitActivityMask',default=None,help="a feature not in use")

	parser.add_argument('--Number_active',type=int,default=2,help="Number of schema actively used for each RNN unit")

	parser.add_argument('--device',type=str,default="cpu",help="device")

	parser.add_argument('--schema_weighting',default=None,
						help="weight for each scheme for each RNN units. dimension (NumUnits , Number_Schemas)")

	parser.add_argument('--num_schemas',type=int,default=4,
						help="Number of schemas, need to be an even number or compatible with h size)")

	parser.add_argument('--StoreResults',type=bool,default=False,
						help="if the evaluation results shoude be recorded into a csv")



	args = parser.parse_args()
	args.mem = args.recurrence > 1
	# Set seed for all randomness sources

	utils.seed(args.seed)

	# Set device

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}\n")

	# Load environments

	envs = []
	for i in range(args.procs):
		env = utils.make_env(args.env, args.seed + 10000 * i)
		envs.append(env)
	env = ParallelEnv(envs)
	obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
	print("Environments loaded\n")

	# Load agent
	
	AllModels=[name for name in os.listdir("storage") if "SWtraning_" in name]
	AllFeatures=[]#X
	SW1x2h=[]#Y1
	SW1h2h=[]#Y2
	###print(AllModels)
	for model_name in AllModels:
		infor=model_name.split("-")[1]
		print(infor)
		Width = float(re.search('EnvW(.*)H', infor).group(1))
		Height=float(re.search('H(.*)T', infor).group(1))
		
		Blocks=[0,0,0]
		Blocks[int(re.search('T(.*)', infor).group(1)[0])]=1
		
		#print(Blocks)
		Targets=re.search('T(.*)', infor).group(1)[1:]
		if Targets=="Goal":
			Targets=[1,0]
		elif Targets=="Objects":
			Targets=[0,1]
		#print(Targets)

		Features=[]
		Features.append(Width/9.0)
		Features.append(Height/9.0)
		Features=Features+Blocks+Targets
		Features=torch.FloatTensor(Features)
		#print(Features)
		AllFeatures.append(Features)


		model_dir = utils.get_model_dir(model_name)
		acmodel = ACModel(obs_space=obs_space, action_space=envs[0].action_space,
					  use_memory=args.mem, use_text=args.text, use_rim =args.use_rim,
					  NUM_UNITS=args.NUM_UNITS,
					  k=args.k, rnn_cell=args.rnn_cell,
					  UnitActivityMask=args.UnitActivityMask,
					  Number_active=args.Number_active, device=device,
					  schema_weighting=args.schema_weighting, num_schemas=args.num_schemas,rim_type=args.rim_type)

		acmodel.load_state_dict(utils.get_model_state(model_dir))
		acmodel.to(device)
		acmodel.eval()

		[print(Keys) for Keys, Values in acmodel.state_dict().items()]
		print(len([Keys for Keys, Values in acmodel.state_dict().items()]))

		
		for Name,Param in acmodel.named_parameters():
			if Name in ["memory_rnn.rnn.x2h.SharedParameters.schema_weighting"]:
				#print(Param.data)
				SW1x2h.append(Param.data.flatten())
			if Name in ["memory_rnn.rnn.h2h.SharedParameters.schema_weighting"]:
				SW1h2h.append(Param.data.flatten())

	print(str(SW1x2h[0].size()))
	print(SW1x2h[0])
	print(SW1x2h[0].flatten())
	print(SW1x2h[0].flatten().reshape(args.NUM_UNITS,args.num_schemas))

	print("y1:")
	print(torch.stack(SW1x2h,0))
	print("y1 size")
	print(torch.stack(SW1x2h,0).size())

	print("y2:")
	print(torch.stack(SW1h2h,0))
	print("y2 size")
	print(torch.stack(SW1h2h,0).size())


	print("Input features:")
	print(torch.stack(AllFeatures,0))
	print("Input feature size")
	print(torch.stack(AllFeatures,0).size())

	AllFeatures=torch.stack(AllFeatures,0)
	SW1x2h=torch.stack(SW1x2h,0)
	SW1h2h=torch.stack(SW1h2h,0)

	print("data loading finished")

			

	print("training model for schema weighting prediction based on task features")
	sampleSize=AllFeatures.size()[0]
	din=AllFeatures.size()[1] 
	dout=SW1x2h.size()[1]
	batch_size=4
	n_hidden=6
	num_epochs=30

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:"+str(device))

	##Shuffle data 	
	from random import shuffle
	Indices=list(range(sampleSize))
	shuffle(Indices)

	AllFeatures=AllFeatures[Indices,:]
	SW1x2h=SW1x2h[Indices,:]
	SW1h2h=SW1h2h[Indices,:]
	



	TrainRatio=0.95
	
	#training model X-> Y1
	Model=SW_net(din,dout,n_hidden)
	X=AllFeatures
	Y=SW1x2h
	PATH="SW_predictor_x2h.pt"

	X_train=X[range(int(TrainRatio*sampleSize)),:]
	Y_train=Y[range(int(TrainRatio*sampleSize)),:]

	X_test=X[range(int(TrainRatio*sampleSize),sampleSize),:]
	Y_test=Y[range(int(TrainRatio*sampleSize),sampleSize),:]
	#print(Model.forward(X))
	#print(Model.forward(X).size())
	#print(Model)
	
	loss_func = nn.MSELoss()  
	optimizer = torch.optim.Adam(Model.parameters())
	
	# train the network
	for epoch in range(num_epochs):
		permutation = torch.randperm(X_train.size()[0])
		Step=0
		Loss=0
		for i in range(0,X_train.size()[0], batch_size):
			
			indices = permutation[i:i+batch_size]
			batch_x=X_train[indices].to(device)
			batch_y=Y_train[indices].to(device)
			prediction = Model(batch_x)     # input x and predict based on x
			loss = loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)
			Loss+=float(loss.data)
			optimizer.zero_grad()   # clear gradients for next train
			loss.backward()         # backpropagation, compute gradients
			optimizer.step()        # apply gradients
		print('Epoch: [%d/%d],Loss: %.4f'%(epoch+1, num_epochs,Loss))
		
	print("training finished")
	torch.save(Model.state_dict(), PATH)

	Predction=Model(X_test)
	MSE=nn.MSELoss()(Predction,Y_test)
	print("testing MAE sqrt: [%.4f]" %(float(MSE.data)**0.5))

	#Model.load_state_dict(torch.load(PATH))

	#training model X-> Y1
	Model=SW_net(din,dout,n_hidden)
	X=AllFeatures
	Y=SW1x2h
	PATH="SW_predictor_h2h.pt"

	X_train=X[range(int(TrainRatio*sampleSize)),:]
	Y_train=Y[range(int(TrainRatio*sampleSize)),:]

	X_test=X[range(int(TrainRatio*sampleSize),sampleSize),:]
	Y_test=Y[range(int(TrainRatio*sampleSize),sampleSize),:]
	#print(Model.forward(X))
	#print(Model.forward(X).size())
	#print(Model)
	
	loss_func = nn.MSELoss()  
	optimizer = torch.optim.Adam(Model.parameters())
	
	# train the network
	for epoch in range(num_epochs):
		permutation = torch.randperm(X_train.size()[0])
		Step=0
		Loss=0
		for i in range(0,X_train.size()[0], batch_size):
			
			indices = permutation[i:i+batch_size]
			batch_x=X_train[indices].to(device)
			batch_y=Y_train[indices].to(device)
			prediction = Model(batch_x)     # input x and predict based on x
			loss = loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)
			Loss+=float(loss.data)
			optimizer.zero_grad()   # clear gradients for next train
			loss.backward()         # backpropagation, compute gradients
			optimizer.step()        # apply gradients
		print('Epoch: [%d/%d],Loss: %.4f'%(epoch+1, num_epochs,Loss))
		
	print("training finished")
	torch.save(Model.state_dict(), PATH)

	Predction=Model(X_test)
	MSE=nn.MSELoss()(Predction,Y_test)
	print("testing MAE sqrt: [%.4f]" %(float(MSE.data)**0.5))



	print("testing SW prediction")
	SW1x2h,SW1h2h=Predict_SW(Width=4,Height=16,BlockType=1,Targets="Objects")
	print(SW1x2h)
	print(SW1h2h)