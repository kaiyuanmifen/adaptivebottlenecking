#!/bin/bash



#####Drone env

declare -a All_Data=('CartPole-v1')


declare -a Methods=("Adaptive_Hierachical" "Adaptive_Quantization" "Quantization" "Original")

declare -a Methods=("Quantization")


Rounds=($(seq 1 1 10))




for data in "${All_Data[@]}"
do
	for Method in "${Methods[@]}"
	do

		for Round in "${Rounds[@]}"
		do 
		sbatch Submit_slurm_train_DeepQ.sh $Method $data $Round

		done
	done
done

