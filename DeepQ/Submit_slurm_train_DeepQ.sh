#!/bin/bash
#SBATCH --job-name=train_DQN_adaptive
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=35G               # memory (per node)
#SBATCH --time=0-5:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate MARL





Method=$1
data=$2
Round=$3

echo "Method $Method"

echo "data $data"

echo "Round: $Round";

python deepQGym.py \
	--Round ${Round} \
	--Method ${Method} \
	--data ${data}




