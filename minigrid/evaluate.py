import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils

from SchemaWeightingPredictor import *


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
					help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
					help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
					help="number of episodes of evaluation (default: 1000)")
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
parser.add_argument('--use_rim',action = 'store_true', default = False,help="whether use RIM")

parser.add_argument('--rim_type',type=str,default="Original",
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

parser.add_argument('--TaskName',type=str,default="NotSpecified",
					help="Name of the task when stored")

parser.add_argument('--PredictSchemaWeights',action = 'store_true', default = False,
					help="Seeing a new task, whether predict the schema weighting first")

parser.add_argument('--Width',type=int,default=9,
					help="Width of the minigrid environments")

parser.add_argument('--Height',type=int,default=9,
					help="Height of the minigrid environments")

parser.add_argument('--BlockType',type=int,default=0,
					help="BlockType of the minigrid environments[0,1,2]")

parser.add_argument('--Targets',type=str,default="Goal",
					help="Targets of the minigrid environments:Goal or Objects")


parser.add_argument('--Method',type=str,default="Original",
					help="Method for quantization")



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
print("Environments loaded\n")


# If schema weight should be predicted 
if (args.use_rim and args.rim_type=='Shared' and args.PredictSchemaWeights):
	print("Predicting Schema weights")
	SW1x2h,SW1h2h=Predict_SW(Width=args.Width,Height=args.Height,
		BlockType=args.BlockType,Targets=args.Targets)
	schema_weighting_pred=[SW1x2h,SW1h2h]
	#print("Schema weights:")
	#print(schema_weighting_pred)
else:
	schema_weighting_pred=None



# # Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(obs_space=env.observation_space, action_space=env.action_space,
					model_dir=model_dir, device=device, argmax=args.argmax,
					num_envs=args.procs, use_memory=args.mem, use_text=args.text,
					use_rim =args.use_rim,rim_type=args.rim_type,
				  NUM_UNITS=args.NUM_UNITS,
				  k=args.k, rnn_cell=args.rnn_cell,
				  UnitActivityMask=args.UnitActivityMask,
				  Number_active=args.Number_active,
				  schema_weighting=args.schema_weighting, num_schemas=args.num_schemas,
				  PredictSchemaWeights=args.PredictSchemaWeights,
				  schema_weighting_to_use=schema_weighting_pred,args=args)
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

while log_done_counter < args.episodes:
	actions = agent.get_actions(obss)
	obss, rewards, dones, _ = env.step(actions)
	agent.analyze_feedbacks(rewards, dones)

	log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
	log_episode_num_frames += torch.ones(args.procs, device=device)

	for i, done in enumerate(dones):
		if done:
			log_done_counter += 1
			logs["return_per_episode"].append(log_episode_return[i].item())
			logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

	mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
	log_episode_return *= mask
	log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
	  .format(num_frames, fps, duration,
			  *return_per_episode.values(),
			  *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
	print("\n{} worst episodes:".format(n))

	indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
	for i in indexes[:n]:
		print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))



n = 20
if n > 0:
	print("\n{} best episodes:".format(n))

	indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k],reverse=True)
	for i in indexes[:n]:
		print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))


#save the evaluation results in a csv file 

	
from csv import writer
def append_list_as_row(file_name, list_of_elem):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		csv_writer = writer(write_obj)
		# Add contents of list as last row in the csv file
		csv_writer.writerow(list_of_elem)



if args.StoreResults:
	
	list_of_elem=["{}".format(args.TaskName),
		"{}".format(num_frames),
		"{:.0f}".format(fps),
		"{}".format(duration),
		"{:.2f} {:.2f} {:.2f} {:.2f}".format(*return_per_episode.values()),
		"{:.1f} {:.1f} {} {}".format(*num_frames_per_episode.values())]

	append_list_as_row("ExperimentalResults.csv", list_of_elem)