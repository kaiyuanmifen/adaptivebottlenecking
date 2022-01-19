#!/bin/bash



#####Drone env

declare -a All_Data=('CartPole-v1' "LunarLander-v2")

#declare -a Methods=("Adaptive_Hierachical" "Adaptive_Quantization" "Quantization" "Original")

declare -a Methods=("Quantization")

# declare -a All_Data=('AirRaid-v0')

# declare -a Methods=("Quantization")

Rounds=($(seq 1 1 5))

# declare -a All_Data=("LunarLander-v2")

# declare -a Methods=("Original")

# declare -a Methods=("Quantization")


# declare -a All_hypotheses=(5.2)



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

