#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-9:00
#SBATCH --mem=15G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module load gcc/6.2.0 cuda/10.1
module load python/3.7.4
source /home/dl249/RIM/MiniGrid_RIM_Env_New/bin/activate

######################codes above this line are for clusters####################
echo "this code true weight of different schemas while freezing shared parameters"

Envs=$1

echo "Env:${Envs}"

NumberOfSteps=$2

echo "NumberOfSteps:${NumberOfSteps}"

python3 train.py --algo ppo --env $Envs --model DLEnvs_RIMSP --frames $NumberOfSteps --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --model_dir_save "storage/SWtraning_${Envs}" --freeze_sharedParameters
echo "training finished"
