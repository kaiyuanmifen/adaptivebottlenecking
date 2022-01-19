#!/bin/bash 

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
    			echo "${i} ${j} ${k} ${l}"
    			python3 DianboEinvironmentsSWtraining.py --Width "$i" --Height "$j" --FinalGoal "$k" --NothingOrWallOrLava "$l"
			done
		done 
	done 
done


echo "registeration finished"