import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys

import utils
from model import ACModel
from QuantizerFunction import QuantizerFunction

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
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

parser.add_argument('--schema_weighting',default=None,
                    help="weight for each scheme for each RNN units. dimension (NumUnits , Number_Schemas)")

parser.add_argument('--num_schemas',type=int,default=4,
                    help="Number of schemas, need to be an even number or compatible with h size)")

###in some experiments the newly trained model needs to be saved at a different dir than the original model
parser.add_argument('--model_dir_save',type=str,default=None,
                    help="when this value is None, model will be saved at the original dir(model_dir)")


parser.add_argument('--freeze_sharedParameters',action = 'store_true', default = False,help="whether the shared parameters in RIMSP should be freezed")



parser.add_argument('--Method',type=str,default="Original",
                    help="Method for quantization")



args = parser.parse_args()

args.mem = args.recurrence > 1

if bool(args.use_rim):
    print("using RIM")





# Set run dir
def reshape_reward(obs, action, reward, done):
    if not done:
        reward = -1
    else:
        reward = 1
    return reward

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"


model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)
model_dir=model_dir+str(args.seed)

# Load loggers and Tensorboard writer
if args.model_dir_save is None:
	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)
	tb_writer = tensorboardX.SummaryWriter(model_dir)
if args.model_dir_save is not None:
	txt_logger = utils.get_txt_logger(args.model_dir_save)
	csv_file, csv_logger = utils.get_csv_logger(args.model_dir_save)
	tb_writer = tensorboardX.SummaryWriter(args.model_dir_save)
# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

print("obs_space")
print(obs_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(args=args,obs_space=obs_space, action_space=envs[0].action_space,
                  use_memory=args.mem, use_text=args.text, use_rim =args.use_rim,
                  NUM_UNITS=args.NUM_UNITS,
                  k=args.k, rnn_cell=args.rnn_cell,
                  UnitActivityMask=args.UnitActivityMask,
                  Number_active=args.Number_active, device=device,
                  schema_weighting=args.schema_weighting, num_schemas=args.num_schemas,rim_type=args.rim_type)


#check if the parameter were mapped correctly especially when using on GPU
[print(Keys) for Keys, Values in acmodel.state_dict().items()]
print(len([Keys for Keys, Values in acmodel.state_dict().items()]))

txt_logger.info("number of model parameters:"+str(sum([param.nelement() for param in acmodel.parameters()])) +"\n")

if args.freeze_sharedParameters:
    print("freezing shared parameters")
    for Name,Param in acmodel.named_parameters():
        if Name in ["memory_rnn.rnn.x2h.SharedParameters.all_weight",
                        "memory_rnn.rnn.x2h.SharedParameters.all_bias",
                        "memory_rnn.rnn.h2h.SharedParameters.all_weight",
                        "memory_rnn.rnn.h2h.SharedParameters.all_bias"]:
            Param.requires_grad=False


    print("Freezed parameter sets")
    for Name,Param in acmodel.named_parameters():
        if Param.requires_grad==False:
            print(Name)
if args.freeze_sharedParameters==False:
    print("NOT freezing shared parameters")  
        

    #print(param.requires_grad)


if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# # Load algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        if args.model_dir_save is None:
        	utils.save_status(status, model_dir)
        if args.model_dir_save is not None:
        	utils.save_status(status, args.model_dir_save)

        txt_logger.info("Status saved")
