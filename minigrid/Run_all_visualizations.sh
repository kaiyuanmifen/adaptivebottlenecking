#!/bin/bash 

for TrainingEnv in "DLEnvs" "MiniGrid-Empty-5x5-v0" "MiniGrid-DoorKey-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0"
do	
	for TestingEnv in "MiniGrid-DLEnv-random-v0" "MiniGrid-Empty-5x5-v0" "MiniGrid-DoorKey-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0" "MiniGrid-Empty-16x16-v0" "MiniGrid-DistShift2-v0" "MiniGrid-DoorKey-16x16-v0" "MiniGrid-MultiRoom-N6-v0" "MiniGrid-Empty-Random-9x9-v0" "MiniGrid-DLTestEnvS7N5BoxGoal-random-v0" "MiniGrid-DLTestEnvS7N5KeyGoal-random-v0"
	do
		python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_LSTM" --seed 8 --gif "${TrainingEnv}_${TestingEnv}_LSTM" --episodes 3 
		python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIM" --seed 8 --gif "${TrainingEnv}_${TestingEnv}_RIM"  --use_rim --rim_type Original --NUM_UNITS 4 --k 3 --episodes 3
		python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 8 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 3
	done
done
