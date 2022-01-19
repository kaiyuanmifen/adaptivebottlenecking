#!/bin/bash
#SBATCH --job-name=train_adap
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=40G               # memory (per node)
#SBATCH --time=0-5:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate MADDPG




Method=$1


Round=$2



echo "Method $Method"

echo "Round $Round"

python main.py --Method $Method --Round $Round


