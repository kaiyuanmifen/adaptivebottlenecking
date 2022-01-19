#!/bin/bash


echo "check performance of different quantization target" 



ExperimentName="AdaptiveQuantization"

declare -a Quantization_Targets=("edge_sum")


declare -a RatioDataForTrainings=("0001")


# # ####2D shapes
declare -a TargetTasks=("shapes_eval" "shapesN3" "shapesN4" "shapesN2")
#declare -a TargetTasks=("shapes_eval")
 #"shapesN7" "shapesN10"
declare -a SourceModels=("shapes")



for SourceModel in "${SourceModels[@]}"
do
	for TargetTask in "${TargetTasks[@]}"
	do
		for Quantization_target in "${Quantization_Targets[@]}"
		do	
			for Ratio in "${RatioDataForTrainings[@]}"
			do
				echo $SourceModel
				echo $TargetTask
				echo $Quantization_target
				echo $Ratio
				sbatch Submit_slurm_job_evaluation_allGL.sh $SourceModel $TargetTask $Quantization_target $Ratio $ExperimentName
			done
		done
	done
done





#######different Atari games


#declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")

declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")


declare -a TargetTasks=("_eval" "_eval_OOD2")

#declare -a TargetTasks=("_eval" "_eval_OOD" "_eval_OOD2")

declare -a Quantization_Targets=("edge_sum")

declare -a RatioDataForTraining=(0.001)


#declare -a SourceModels=("spaceinvaders")


for Game in "${Games[@]}"
do
	echo ${Game}

	for TargetTask in "${TargetTasks[@]}"
	do

		for Quantization_target in "${Quantization_Targets[@]}"
		do

			for Ratio in "${RatioDataForTraining[@]}"
			do
				
				echo $SourceModel
				echo $TargetTask
				echo $Quantization_target
				echo $Ratio
				sbatch Submit_slurm_job_evaluation_allGL_Atari.sh "$Game" "${Game}${TargetTask}" $Quantization_target $Ratio $ExperimentName
			done
		done
	done
done




