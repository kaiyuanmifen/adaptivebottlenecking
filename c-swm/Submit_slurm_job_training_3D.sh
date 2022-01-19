#!/bin/bash
#SBATCH --job-name=training
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=40G               # memory (per node)
#SBATCH --time=0-02:20            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate GNN




Task="cubes"

Quantization=${1}

n_codebook_embedding=${2}

Quantization_method=${3}

n_quuantization_segments=${4}

seed=${5}

Quantization_target=${6}

RatioDataForTraining=${7}



if [ "$Quantization" = true ] 
then
	echo "conducting quantization"
	
	name=${Task}"_"${Quantization}"_"${n_codebook_embedding}"_"${Quantization_method}"_"${n_quuantization_segments}"_"${Quantization_target}"_"${RatioDataForTraining}"_"${seed}

	name="${name//./}"
	echo Running version $name

	python train.py --dataset data/cubes_train.h5 --encoder large --name ${name} --Quantization --n_codebook_embedding ${n_codebook_embedding} --Quantization_method ${Quantization_method} --n_quuantization_segments ${n_quuantization_segments}  --Quantization_target ${Quantization_target} --RatioDataForTraining ${RatioDataForTraining} --seed ${seed}
else
	echo "not conducting quantization"
	name=${Task}"_orignal_"${RatioDataForTraining}"_"${seed}

	name="${name//./}"
	echo Running version $name

	python train.py --dataset data/cubes_train.h5 --encoder large --name ${name} --RatioDataForTraining ${RatioDataForTraining} --seed ${seed}

fi
