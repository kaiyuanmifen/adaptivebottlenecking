#!/bin/bash 

#remove old experimental results
rm ExperimentalResults.csv

echo "Evaluate LSTM/RIM/RIMSP on DLEnvs different environments"

# ./ExperimentalEvaluation.sh DLEnvs_LSTM false Original false
# ./ExperimentalEvaluation.sh DLEnvs_RIM true Original false
# ./ExperimentalEvaluation.sh DLEnvs_RIMSP true Shared false
# ./ExperimentalEvaluation.sh DLEnvs_RIMSP true Shared true

sbatch ExperimentalEvaluation.sh DLEnvs_LSTM false Original false
sbatch ExperimentalEvaluation.sh DLEnvs_RIM true Original false
sbatch ExperimentalEvaluation.sh DLEnvs_RIMSP true Shared true


# echo "Evaluate LSTM/RIM/RIMSP trained on environent in original RIM paper on different environments"

# sbatch ExperimentalEvaluation.sh MiniGrid-Empty-5x5-v0_LSTM false Original
# sbatch ExperimentalEvaluation.sh MiniGrid-Empty-5x5-v0_RIM  true Original
# sbatch ExperimentalEvaluation.sh MiniGrid-Empty-5x5-v0_RIMSP true Shared

# sbatch ExperimentalEvaluation.sh MiniGrid-MultiRoom-N2-S4-V0_LSTM false Original
# sbatch ExperimentalEvaluation.sh MiniGrid-MultiRoom-N2-S4-V0_RIM true Original
# sbatch ExperimentalEvaluation.sh MiniGrid-MultiRoom-N2-S4-V0_RIMSP true Shared

# sbatch ExperimentalEvaluation.sh MiniGrid-DoorKey-5x5-v0_LSTM false Original
# sbatch ExperimentalEvaluation.sh MiniGrid-DoorKey-5x5-v0_RIM true Original
# sbatch ExperimentalEvaluation.sh MiniGrid-DoorKey-5x5-v0_RIMSP true Shared