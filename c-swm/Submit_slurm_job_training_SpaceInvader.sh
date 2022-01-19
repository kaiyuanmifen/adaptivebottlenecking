#!/bin/bash
#SBATCH --job-name=GNN
#SBATCH --ntasks=1
#SBATCH --time=0:45:00
#SBATCH --mem=20Gb
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -p gpu_quad


###########cluster information above this line


###load environment 
# module load anaconda/3
# module load cuda/10.1
# conda activate GNN


module load gcc/6.2.0 cuda/10.1
module load python/3.7.4

#virtualenv GNN
source /home/dl249/GNN/GNN/bin/activate



Task="spaceinvaders"

Quantization=${1}

n_codebook_embedding=${2}

Quantization_method=${3}

n_quuantization_segments=${4}

seed=${5}

Quantization_target=${6}

RatioDataForTraining=${7}




if [ "$Quantization" = true ] ;
	then
		echo "conducting quantization"
		
		name=${Task}"_"${Quantization}"_"${n_codebook_embedding}"_"${Quantization_method}"_"${n_quuantization_segments}"_"${Quantization_target}"_"${RatioDataForTraining}"_"${seed}

		name="${name//./}"
		echo Running version $name

		python train.py --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 400 --name ${name} --Quantization --n_codebook_embedding ${n_codebook_embedding} --Quantization_method ${Quantization_method} --n_quuantization_segments ${n_quuantization_segments}  --Quantization_target ${Quantization_target} --RatioDataForTraining ${RatioDataForTraining} --seed ${seed}
	else
		echo "not conducting quantization"
		name=${Task}"_orignal_"${RatioDataForTraining}"_"${seed}

		name="${name//./}"
		echo Running version $name

		python train.py --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 400 --name ${name} --RatioDataForTraining ${RatioDataForTraining} --seed ${seed}


	fi
