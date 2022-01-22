#!/bin/bash 

echo "this code submit all trainig jobs for GNN tasks"



# ######stage 1 submit jobs to see whihich target of quantization works best for jobs


declare -a CodeBookSizeS=(16)
declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")
#declare -a Quantization_methodS=("Adaptive_Hierachical")

declare -a n_segments=(1)
declare -a seedS=($(seq 1 1 10))
#declare -a Quantization_Targets=("edge" "edge_sum" "node_action" "node_final" "node_initial")
declare -a Quantization_Targets=("edge_sum")
declare -a RatioDataForTraining=(0.001)



# ###2D shapes

for r in "${seedS[@]}"
do
	
	for s in "${CodeBookSizeS[@]}" 
	do		
		for m in "${Quantization_methodS[@]}"
		do

			for n in "${n_segments[@]}"
			do 
				for t in "${Quantization_Targets[@]}"
				do
					for p in "${RatioDataForTraining[@]}"
					do
						echo "$r"
						echo "$j"
						echo "$p"
						echo "$l"
						echo "$s"
						echo '$t'
						#sbatch Submit_slurm_job_training_2D.sh false 0 None 0 $r $t $p
						sbatch Submit_slurm_job_training_2D.sh true $s $m $n $r $t $p
					done
				done
			done
		done
	done 
done




# ##artari games


declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")
declare -a CodeBookSizeS=(16)
declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")

declare -a n_segments=(1)
declare -a seedS=($(seq 1 1 10))
#declare -a Quantization_Targets=("edge" "edge_sum" "node_action" "node_final" "node_initial")
declare -a Quantization_Targets=("edge_sum")
declare -a RatioDataForTraining=(0.001)




for Game in "${Games[@]}"
do 

	# for r in "${seedS[@]}"
	# do
	# 	for p in "${RatioDataForTraining[@]}"
	# 	do
	# 		sbatch Submit_slurm_job_training_AtariGames.sh false 0 None 0 $r None $p ${Game}
	# 	done
	# done


	for r in "${seedS[@]}"
	do
		for s in "${CodeBookSizeS[@]}" 
		do		
			for m in "${Quantization_methodS[@]}"
			do

				for n in "${n_segments[@]}"
				do 
					for t in "${Quantization_Targets[@]}"
					do
						for p in "${RatioDataForTraining[@]}"
						do
							echo "$r"
							echo "$s"
							echo "$m"
							echo "$n"
							echo "$t"
							echo "$p"
							sbatch Submit_slurm_job_training_AtariGames.sh true $s $m $n $r $t $p ${Game}
						done
					done
				done
			done
		done 
	done
done

