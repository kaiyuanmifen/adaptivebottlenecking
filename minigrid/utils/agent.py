import torch

import utils
from model import ACModel


class Agent:
	"""An agent.

	It is able:
	- to choose an action given an observation,
	- to analyze the feedback (i.e. reward and done state) of its action."""

	def __init__(self, obs_space, action_space, model_dir, device=None, argmax=False, num_envs=1, use_memory=True,
				 use_text=False, use_rim = False,rim_type="Original",NUM_UNITS=4,
				 k=3,rnn_cell="LSTM",UnitActivityMask=None, Number_active=3,
				 schema_weighting=None, num_schemas=8,
				 PredictSchemaWeights=None,schema_weighting_to_use=None,args=None):
		obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
		#self.acmodel = ACModel(obs_space, action_space, use_rim = use_rim)
		print("rim type:"+rim_type)
		self.acmodel =ACModel(args=args,obs_space=obs_space, action_space=action_space,
				use_memory=use_memory, use_text=use_text,
				use_rim=use_rim,rim_type=rim_type,
				NUM_UNITS=NUM_UNITS,k=k, rnn_cell=rnn_cell,
				UnitActivityMask=UnitActivityMask,
				Number_active=Number_active, device=device,
				schema_weighting=schema_weighting, num_schemas=num_schemas)
		self.device = device
		self.argmax = argmax
		self.num_envs = num_envs

		


		if self.acmodel.recurrent:
			self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size).to(device)
		
		self.acmodel.load_state_dict(utils.get_model_state(model_dir))

		

		###Assign predicted Schema weights to the model if needed
		if PredictSchemaWeights: 
			State_dict=self.acmodel.state_dict()
			if rnn_cell == "LSTM":
				print("Using predicted schema_weighting for LSTM")
				#print(schema_weighting_to_use)
				State_dict["memory_rnn.rnn.x2h.SharedParameters.schema_weighting"]=schema_weighting_to_use[0].to(self.device)
				State_dict["memory_rnn.rnn.h2h.SharedParameters.schema_weighting"]=schema_weighting_to_use[1].to(self.device)
			if rnn_cell == "GRU":
				print("Using predicted schema_weighting for gru(To be done)")

			self.acmodel.load_state_dict(State_dict)

			###check if the tensor assigment was correct 
			
			for name, param in self.acmodel.named_parameters():
				if (param.requires_grad) and (name=="memory_rnn.rnn.x2h.SharedParameters.schema_weighting") :
					
					SW1=torch.all(torch.eq(param.data.to(self.device), schema_weighting_to_use[0].to(self.device)))
					
			
				if (param.requires_grad) and (name=="memory_rnn.rnn.h2h.SharedParameters.schema_weighting") :
					
					SW2=torch.all(torch.eq(param.data.to(self.device), schema_weighting_to_use[1].to(self.device)))

			if SW1 and SW2:
				print ("Schema weighting assigned correctly")
			else:
				print ("Schema weighting assigned WRONGLY")


		
		self.acmodel.to(self.device)
		self.acmodel.eval()
		if hasattr(self.preprocess_obss, "vocab"):
			self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

	def get_actions(self, obss):
		preprocessed_obss = self.preprocess_obss(obss, device=self.device)

		with torch.no_grad():
			if self.acmodel.recurrent:
				dist, _, self.memories,ExtraLoss = self.acmodel(preprocessed_obss, self.memories)
			else:
				dist, _ = self.acmodel(preprocessed_obss)

		if self.argmax:
			actions = dist.probs.max(1, keepdim=True)[1]
		else:
			actions = dist.sample()

		return actions.cpu().numpy()

	def get_action(self, obs):
		return self.get_actions([obs])[0]

	def analyze_feedbacks(self, rewards, dones):
		if self.acmodel.recurrent:
			masks = 1 - torch.tensor(dones, dtype=torch.float).to(self.device).unsqueeze(1)
			self.memories *= masks

	def analyze_feedback(self, reward, done):
		return self.analyze_feedbacks([reward], [done])
