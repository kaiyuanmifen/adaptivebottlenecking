#!/bin/bash 


echo "this code true weight of different schemas while freezing shared parameters"

###this is number has already included the original number of training frames, eg. if it is 2e6 , it actually means 0.5e6 (orginal training got 1.5e6 frames)
NumberOfSteps=2000000


declare -a Widths=(5 6 7 8 9)
declare -a Heights=(5 6 7 8 9)
declare -a FinalGoals=("Objects" "Goal")
declare -a NothingOrWallOrLava=(0 1 2)

for i in "${Widths[@]}"
do
	for j in "${Heights[@]}"
    do
    	for k in "${FinalGoals[@]}"
    	do
    		for l in "${NothingOrWallOrLava[@]}"
    		do 
    			Envs="MiniGrid-DLSWEnvW${i}H${j}T${l}${k}-random-v0"
    			echo "$Envs"
    			
				sbatch train_schema_weighting.sh $Envs $NumberOfSteps

			done
		done 
	done 
done


echo "SW training jobsubmission finished"
