#!/bin/bash
#SBATCH --job-name=train_adap
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=35G               # memory (per node)
#SBATCH --time=0-3:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate MARL




data=$1

N_agents=$2

Method=$3


Round=$4

echo "data: $data";

echo "N_agents $N_agents"

echo "Method $Method"

echo "Round $Round"


echo "Round: $Round";
CUDA_LAUNCH_BLOCKING=1 python PPO_test_FindBox_AdaptiveBottlenecking.py \
						--Round ${Round} \
						--data ${data} \
						--N_agents ${N_agents} \
						--Method ${Method}



