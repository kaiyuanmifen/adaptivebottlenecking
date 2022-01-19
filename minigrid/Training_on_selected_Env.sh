#!/bin/bash
#SBATCH --job-name=training_minigrid
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=30G               # memory (per node)
#SBATCH --time=0-03:20            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate GYMEnv





######################codes above this line are for clusters####################
#Experiemnt setting 
#ModelName="DLEnvs_LSTM"
ModelName="$1"


echo "ModelName: $ModelName"

#use_RIM=false
use_RIM=$2

#rim_type="Shared"
#rim_type="Original"
rim_type="$3"

#record how many frames have been done
#num_frames=1000000

num_frames=$4
echo "Numner of frames:$num_frames"

Env=$5
echo "training Env:$Env"

Method=$6

echo "Method $Method"
Round=$7
echo "Round $Round"
#rim_type="Shared"
if [ "$use_RIM" = true ] ; 
then
    	echo 'using RIM'
    	echo "RIM type: $rim_type"
else 
		echo 'Not using RIM'
fi


#get a random see 
#RandomSeed=$(( ( RANDOM % 10 )  + 1 ))
#echo "Random seed: $RandomSeed"

#Run the training 

echo "working on $Env"
if [ "$use_RIM" = true ] ; 
then
	echo 'using RIM'
	echo "RIM type: $rim_type"
	python3 train.py --algo ppo --env "$Env" --model "$ModelName" --frames $num_frames --use_rim --rim_type "$rim_type" --NUM_UNITS 8 --k 6 --Number_active 2 --num_schemas 8 --seed $Round --Method $Method
	
else
	echo "Not using RIM"
	python3 train.py --algo ppo --env "$Env" --model "$ModelName" --frames $num_frames --seed $Round --Method $Method

fi
echo "total number of frames done: $num_frames"
echo "finish"


