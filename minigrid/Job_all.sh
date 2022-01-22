#!/bin/bash 

NumberOfSteps=1500000

declare -a All_Data=('MiniGrid-DLEnv-random-v0' "MiniGrid-Empty-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0" "MiniGrid-DoorKey-5x5-v0")

declare -a Methods=("Adaptive_Hierachical" "Adaptive_Quantization" "Quantization" "Original")



# declare -a All_Data=('MiniGrid-DLEnv-random-v0')

# declare -a Methods=("Adaptive_Hierachical" )

# declare -a All_Data=('MiniGrid-DLEnv-random-v0')

# declare -a Methods=("Original")


declare -a Rounds=($(seq 1 1 10))


for data in "${All_Data[@]}"
do
	for Method in "${Methods[@]}"
	do

		for Round in "${Rounds[@]}"
		do 

			Name=${data}"_"${Method}"_"${Round}"_"${Round}
			echo "$Name"

			./Training_on_selected_Env.sh $Name false Original $NumberOfSteps $data $Method $Round


		done
	done
done




####evaluation




# declare -a All_Data_train=('MiniGrid-DLEnv-random-v0' "MiniGrid-Empty-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0" "MiniGrid-DoorKey-5x5-v0")

# declare -a Methods=("Adaptive_Hierachical" "Adaptive_Quantization" "Quantization" "Original")


# #declare -a All_Data=('MiniGrid-DLEnv-random-v0')

# #declare -a Methods=("Adaptive_Hierachical")


# declare -a Rounds=($(seq 1 1 5))


# for data in "${All_Data_train[@]}"
# do
# 	for Method in "${Methods[@]}"
# 	do

# 		for Round in "${Rounds[@]}"
# 		do 

# 			ModelName=${data}"_"${Method}"_"${Round}"_"${Round}
# 			echo "$ModelName"

# 			sbatch ExperimentalEvaluation.sh $ModelName false Original false $Method


# 		done
# 	done
# done

