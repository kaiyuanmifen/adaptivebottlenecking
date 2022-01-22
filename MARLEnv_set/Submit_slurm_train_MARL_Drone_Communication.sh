#!/bin/bash
#SBATCH --job-name=train_adap
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=35G               # memory (per node)
#SBATCH --time=0-3:30            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate MARL




data=$1

N_agents=$2

Method=$3


#Round=$4

echo "data: $data";

echo "N_agents $N_agents"

echo "Method $Method"

Rounds=($(seq 1 1 10))



for Round in "${Rounds[@]}"
do 

echo "Round $Round"

CUDA_LAUNCH_BLOCKING=1 python PPO_Drone_Communication.py \
						--Round ${Round} \
						--data ${data} \
						--N_agents ${N_agents} \
						--Method ${Method}

done

