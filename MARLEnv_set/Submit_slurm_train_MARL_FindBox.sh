#!/bin/bash
#SBATCH --job-name=train_MARL_attentionSchema
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=30G               # memory (per node)
#SBATCH --time=0-2:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate MARL




data=$1

N_agents=$2

hypothesis=$3

Round=$4

echo "data: $data";

echo "N_agents $N_agents"

echo "hypothesis $hypothesis"

echo "Round: $Round";
CUDA_LAUNCH_BLOCKING=1 python PPO_test_FindBox.py \
						--Round ${Round} \
						--data ${data} \
						--N_agents ${N_agents} \
						--hypothesis ${hypothesis}



