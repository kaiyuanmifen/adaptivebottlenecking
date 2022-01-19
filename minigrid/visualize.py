import argparse
import time
import numpy
import torch

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default='N4_RIM',
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1,
                    help="number of episodes to visualize")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
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


args = parser.parse_args()

args.mem = args.recurrence > 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(obs_space=env.observation_space, action_space=env.action_space,
                    model_dir=model_dir, device=device, argmax=args.argmax,
                    num_envs=args.procs, use_memory=args.mem, use_text=args.text,
                    use_rim =args.use_rim,rim_type=args.rim_type,
                  NUM_UNITS=args.NUM_UNITS,
                  k=args.k, rnn_cell=args.rnn_cell,
                  UnitActivityMask=args.UnitActivityMask,
                  Number_active=args.Number_active,
                  schema_weighting=args.schema_weighting, num_schemas=args.num_schemas)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()
    done2 = False
    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
            

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            if episode == 4:
            	done2 = True
            break
    if done2 == True:
    	env.close()
    	break
    #if env.window.closed:
    #    break
print('doneeee')
if args.gif:
    print("Saving gif... ", end="")
    from pathlib import Path
    Path("../figures/demos").mkdir(parents=True, exist_ok=True)

    write_gif(numpy.array(frames), "../figures/demos/"+args.gif+".gif")
    print("Done.")
