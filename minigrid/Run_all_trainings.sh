#!/bin/bash 

NumberOfSteps=1500000

sbatch Training_on_selected_Env.sh DLEnvs_LSTM false Original $NumberOfSteps MiniGrid-DLEnv-random-v0

sbatch Training_on_selected_Env.sh DLEnvs_RIM true Original $NumberOfSteps MiniGrid-DLEnv-random-v0

sbatch Training_on_selected_Env.sh DLEnvs_RIMSP true Shared $NumberOfSteps MiniGrid-DLEnv-random-v0


sbatch Training_on_selected_Env.sh MiniGrid-Empty-5x5-v0_LSTM false Original $NumberOfSteps MiniGrid-Empty-5x5-v0

sbatch Training_on_selected_Env.sh MiniGrid-Empty-5x5-v0_RIM true Original $NumberOfSteps MiniGrid-Empty-5x5-v0

sbatch Training_on_selected_Env.sh MiniGrid-Empty-5x5-v0_RIMSP true Shared $NumberOfSteps MiniGrid-Empty-5x5-v0


sbatch Training_on_selected_Env.sh MiniGrid-MultiRoom-N2-S4-V0_LSTM false Original $NumberOfSteps MiniGrid-MultiRoom-N2-S4-v0

sbatch Training_on_selected_Env.sh MiniGrid-MultiRoom-N2-S4-V0_RIM true Original $NumberOfSteps MiniGrid-MultiRoom-N2-S4-v0

sbatch Training_on_selected_Env.sh MiniGrid-MultiRoom-N2-S4-V0_RIMSP true Shared $NumberOfSteps MiniGrid-MultiRoom-N2-S4-v0



sbatch Training_on_selected_Env.sh MiniGrid-DoorKey-5x5-v0_LSTM false Original $NumberOfSteps MiniGrid-DoorKey-5x5-v0

sbatch Training_on_selected_Env.sh MiniGrid-DoorKey-5x5-v0_RIM true Original $NumberOfSteps MiniGrid-DoorKey-5x5-v0

sbatch Training_on_selected_Env.sh MiniGrid-DoorKey-5x5-v0_RIMSP true Shared $NumberOfSteps MiniGrid-DoorKey-5x5-v0


