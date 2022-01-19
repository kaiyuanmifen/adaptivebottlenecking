#!/bin/bash
#SBATCH --job-name=eval_minigrid
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=30G               # memory (per node)
#SBATCH --time=0-02:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate GYMEnv




######################codes above this line are for clusters####################

echo "Evaluate on different environments"

#a wide range of enviromnemts
declare -a TestingEnvironments=('MiniGrid-DLEnv-random-v0' 
	'MiniGrid-DLTestingW9H16T0Goal-random-v0' 'MiniGrid-DLTestingW16H9T0Goal-random-v0' 
	'MiniGrid-DLTestingW16H16T0Goal-random-v0' 'MiniGrid-DLTestingW9H9T1Goal-random-v0' 
	'MiniGrid-DLTestingW9H9T2Goal-random-v0' 'MiniGrid-DLTestingW9H9T0Objects-random-v0' 
	'MiniGrid-DLTestingW9H9T1Objects-random-v0' 'MiniGrid-DLTestingW9H9T2Objects-random-v0' 
	'MiniGrid-DLTestingW16H16T1Objects-random-v0' 'MiniGrid-DLTestingW16H16T2Objects-random-v0' 
	'MiniGrid-Empty-Random-5x5-v0'	
	'MiniGrid-Empty-16x16-v0'
	'MiniGrid-MultiRoom-N2-S4-v0' 'MiniGrid-MultiRoom-N2-S5-v0' 'MiniGrid-MultiRoom-N6-v0'  
	'MiniGrid-UnlockPickup-v0'
    "MiniGrid-ObstructedMaze-1Dlh-v0" "MiniGrid-ObstructedMaze-1Dlhb-v0"
    'MiniGrid-DistShift1-v0' 'MiniGrid-DistShift2-v0' 
    'MiniGrid-DoorKey-5x5-v0' 'MiniGrid-DoorKey-16x16-v0'
    'MiniGrid-LockedRoom-v0')

#declare -a TestingEnvironments=('MiniGrid-DLEnv-random-v0')

#'MiniGrid-Dynamic-Obstacles-5x5-v0' 
#	'MiniGrid-Dynamic-Obstacles-Random-5x5-v0' 'MiniGrid-Dynamic-Obstacles-6x6-v0' 
#	'MiniGrid-Dynamic-Obstacles-Random-6x6-v0' 
#	'MiniGrid-Dynamic-Obstacles-8x8-v0' 'MiniGrid-Dynamic-Obstacles-16x16-v0'


echo "Testing Environment used"
for i in "${TestingEnvironments[@]}"
do
	echo "$i"
done


#ModelName="CurriculumLearning_LSTM"
#ModelName="CurriculumLearning_RIM"
#ModelName="CurriculumLearning_RIMSP"


ModelName="$1"


echo "ModelName: $ModelName"

#use_RIM=false
use_RIM=$2

#rim_type="Shared"
#rim_type="Original"
rim_type="$3"


PredictSW=$4


Method=$5

RandomSeed=$(( ( RANDOM % 10 )  + 1 ))
echo "Random seed: $RandomSeed"





for i in "${TestingEnvironments[@]}"
do
	echo "working on $i"
	if [ "$use_RIM" = true ] ; 
	then
    	echo 'using RIM'
    	echo "RIM type: $rim_type"
    	python3 evaluate.py --env  "$i" --model "$ModelName" --use_rim --rim_type $rim_type --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}" --seed $RandomSeed --Method ${Method}
	
	else
		echo "Not using RIM"
		python3 evaluate.py --env  "$i" --model "$ModelName" --StoreResults True --TaskName "${i}_${ModelName}" --seed $RandomSeed  --Method ${Method}
	fi
done





#####when schema weighting is predicted from envs features 
if [ $use_RIM ] && [ "$rim_type" == "Shared" ] && [ PredictSW ]
then
	echo "runing on predicted SW"

	i="MiniGrid-DLEnv-random-v0"
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 8 --Height 8 --BlockType 0 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 

	i='MiniGrid-DLTestingW9H16T0Goal-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 16 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW16H9T0Goal-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 9 --BlockType 1 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 

	i='MiniGrid-DLTestingW16H16T0Goal-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 16 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW9H9T1Goal-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 9 --BlockType 1 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 




	i='MiniGrid-DLTestingW9H9T2Goal-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 9 --BlockType 2 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW9H9T0Objects-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 9 --BlockType 0 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW9H9T1Objects-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 9 --BlockType 1 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW9H9T2Objects-random-v0' 
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 9 --BlockType 2 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW16H16T1Objects-random-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 16 --BlockType 1 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DLTestingW16H16T2Objects-random-v0' 
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 16 --BlockType 2 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-Empty-Random-5x5-v0'	
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 5 --Height 5 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-Empty-16x16-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 16 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-MultiRoom-N2-S4-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 4 --Height 4 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 

	i='MiniGrid-MultiRoom-N2-S5-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 5 --Height 5 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-MultiRoom-N6-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 10 --Height 10 --BlockType 0 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-UnlockPickup-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 6 --Height 6 --BlockType 1 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i="MiniGrid-ObstructedMaze-1Dlh-v0"
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 6 --Height 6 --BlockType 1 --Targets "Objects" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i="MiniGrid-ObstructedMaze-1Dlhb-v0"
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 6 --Height 6 --BlockType 1 --Targets "Objects"  --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DistShift1-v0' 
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 7 --BlockType 2 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DistShift2-v0' 
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 9 --Height 7 --BlockType 2 --Targets "Goal"  --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DoorKey-5x5-v0' 
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 5 --Height 5 --BlockType 1 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-DoorKey-16x16-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 16 --Height 16 --BlockType 1 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


	i='MiniGrid-LockedRoom-v0'
	python3 evaluate.py --env  "$i" --model "$ModelName" --PredictSchemaWeights --Width 19 --Height 19 --BlockType 1 --Targets "Goal" --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults True --TaskName "${i}_${ModelName}_SEPred" --seed $RandomSeed 


fi
echo "Evulation finished"