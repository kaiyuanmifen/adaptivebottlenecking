#!/bin/bash




###Drone

declare -a All_Methods=("Adaptive_Hierachical" "Adaptive_Quantization" "Quantization" "Original")

#declare -a All_Methods=("Adaptive_Hierachical")
Rounds=($(seq 1 1 5))


# declare -a All_Methods=( "Adaptive_Hierachical")

for Round in "${Rounds[@]}"
do 

	for Method in "${All_Methods[@]}"
	do


		sbatch Submit_slurm_train_MARL_MADDPG.sh $Method $Round
	done
done









# ####FindBox


# declare -a All_data=("FindBox")
# declare -a All_N_agents=(2)
# declare -a All_Methods=("Adaptive_Quantization" "Quantization" "Original")
# Rounds=($(seq 1 1 10))

# # declare -a All_data=("FindBox")
# # declare -a All_N_agents=(2)
# # declare -a All_Methods=("Original")
# # Rounds=($(seq 1 1 1))



# # declare -a All_data=("FindBox")
# # declare -a All_N_agents=(2)
# # declare -a All_hypotheses=(1)
# # Rounds=($(seq 1 1 1))

# for Round in "${Rounds[@]}"
# do 
# 	for data in "${All_data[@]}"
# 	do

# 		for N_agents in "${All_N_agents[@]}"
# 		do
# 			for Method in "${All_Methods[@]}"
# 			do


# 				sbatch Submit_slurm_train_MARL_FindBox_AdaptiveBottlenecking.sh $data $N_agents $Method $Round
# 			done
# 		done
# 	done
# done