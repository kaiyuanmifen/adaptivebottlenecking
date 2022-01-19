#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=50G               # memory (per node)
#SBATCH --time=0-04:20            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate GNN



echo "this code submit all evaluation jobs for GNN tasks with differnet G and L"




##shapes
SourceModel=${1}

TargetTask=${2}

Quantization_target=${3}

Ratio=${4}   ###percentage of data used for training


ExperimentName=${5} ###name for experimental set

####all model combinations
#declare -a CodeBookSizeS=(16 64 512)
#declare -a CodeBookSizeS=(1 2 4 8 16 32 48 64 512 1024 2048)
#declare -a Quantization_methodS=("VQVAE" "VQVAE_conditional")
#declare -a n_segments=(1 2 4 8 16 32 64 128)

#declare -a n_segments=(1 8 32 64)


declare -a CodeBookSizeS=(96)
declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")
declare -a n_segments=(1)
declare -a seedS=($(seq 1 1 5))




# ###orignal GNN
# for seed in "${seedS[@]}"
# do

# 		Model=${SourceModel}"_orignal_"${Ratio}"_"${seed}
# 		name=${TargetTask}"_by_"${SourceModel}"_orignal_"${Ratio}"_"${seed}
# 		name="${name//./}"
# 		echo Running version $name

# 		python eval.py --dataset "data/"${TargetTask}".h5" \
# 					 --save-folder "checkpoints/${Model}" \
# 					 --num-steps 1 \
# 					 --name ${name} \
# 					 --TargetTask ${TargetTask} \
# 					 --ExperimentName ${ExperimentName} \
# 					 --load_action_dim

# done

##quatized GNN
for n_codebook_embedding in "${CodeBookSizeS[@]}" 
do		
	for Quantization_method in "${Quantization_methodS[@]}"
	do

		for n_quuantization_segments in "${n_segments[@]}"
		do 

				for seed in "${seedS[@]}"
				do
					Quantization=true
					#Model=${SourceModel}"_"${Quantization}"_"${n_codebook_embedding}"_"${Quantization_method}"_"${n_quuantization_segments}"_"${Quantization_target}"_"${Ratio}"_"${seed}
					Model=${SourceModel}"_"${Quantization}"_"${n_codebook_embedding}"_"${Quantization_method}"_"${n_quuantization_segments}"_"${Quantization_target}"_"0001"_"${seed}
			
					name=${TargetTask}"_by_"${SourceModel}"_"${Quantization}"_"${n_codebook_embedding}"_"${Quantization_method}"_"${n_quuantization_segments}"_"${Quantization_target}"_"${Ratio}"_"${seed}

					name="${name//./}"
					echo Running version $name

					python eval.py --dataset "../../data/"${TargetTask}".h5" \
					--save-folder "checkpoints/${Model}" \
					--num-steps 1 \
					--name ${name} \
					--Quantization \
					--n_codebook_embedding ${n_codebook_embedding} \
					--Quantization_method ${Quantization_method} \
					--n_quuantization_segments ${n_quuantization_segments} \
					--TargetTask ${TargetTask} \
					--Quantization_target ${Quantization_target} \
					--ExperimentName ${ExperimentName} \
					--load_action_dim
				done

		done
	done
done 




