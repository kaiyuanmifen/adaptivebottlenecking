#!/bin/bash 

echo "this code submit all trainig jobs for GNN tasks"



# ######stage 1 submit jobs to see whihich target of quantization works best for jobs


declare -a CodeBookSizeS=(96)
declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")
#declare -a Quantization_methodS=("Quantization")

declare -a n_segments=(1)
declare -a seedS=($(seq 1 1 5))
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




##artari games


declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")
declare -a CodeBookSizeS=(96)
declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")
#declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Hierachical" )

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




# #####3_body physics


# declare -a CodeBookSizeS=(96)
# #declare -a Quantization_methodS=("Original" "Quantization" "Adaptive_Quantization" "Adaptive_Hierachical")
# declare -a Quantization_methodS=("Adaptive_Quantization" )

# declare -a n_segments=(4)
# declare -a seedS=($(seq 1 1 1))
# #declare -a Quantization_Targets=("edge" "edge_sum" "node_action" "node_final" "node_initial")
# declare -a Quantization_Targets=("edge_sum")
# declare -a RatioDataForTraining=(0.001)


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						./Submit_slurm_job_training_balls.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done











# #3D Cube shapes

# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$j"
# 						echo "$p"
# 						echo "$l"
# 						echo "$s"
# 						echo '$t'
# 						sbatch Submit_slurm_job_training_3D.sh false 0 None 0 $r $t $p
# 						sbatch Submit_slurm_job_training_3D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done








# ##3-Body physics 

# for r in "${seedS[@]}"
# do
# 	sbatch Submit_slurm_job_training_balls.sh false 0 None 0 $r
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					echo "$j"
# 					echo "$p"
# 					echo "$l"
# 					echo "$s"
# 					echo '$t'
# 					sbatch Submit_slurm_job_training_balls.sh true $s $m $n $r $t
# 				done
# 			done
# 		done
# 	done 
# done


# echo "job submission done"







# echo "stage 2 training with different with different percentage of data(and test on in-distribution)"



# declare -a CodeBookSizeS=(512)
# declare -a Quantization_methodS=("VQVAE")
# declare -a n_segments=(4)
# declare -a seedS=(1 2 3 4 5)
# declare -a Quantization_Targets=("edge_sum")
# declare -a RatioDataForTraining=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.6 1.0)






# ############2D shapes


# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_2D.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_2D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done




# #3D Cube shapes

# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_3D.sh false 0 None 0 $r None $p
# 	done
# done

# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do	
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_3D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done








# ##3-Body physics 

# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 			sbatch Submit_slurm_job_training_balls.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_balls.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done



# ##########Atari Pong
# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_AtariPong.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_AtariPong.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done





# #####################Space Invader
# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_SpaceInvader.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_SpaceInvader.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done






##########additional Atari games 


# declare -a Games=("Boxing" "DoubleDunk" "MsPacman")

# for Game in "${Games[@]}"
# do
# 	echo ${Game}



# 	for r in "${seedS[@]}"
# 	do
# 		for p in "${RatioDataForTraining[@]}"
# 		do
# 			sbatch Submit_slurm_job_training_Atari_additionalGames.sh false 0 None 0 $r None $p ${Game}
# 		done
# 	done


# 	for r in "${seedS[@]}"
# 	do
# 		for s in "${CodeBookSizeS[@]}" 
# 		do		
# 			for m in "${Quantization_methodS[@]}"
# 			do

# 				for n in "${n_segments[@]}"
# 				do 
# 					for t in "${Quantization_Targets[@]}"
# 					do
# 						for p in "${RatioDataForTraining[@]}"
# 						do
# 							echo "$r"
# 							echo "$s"
# 							echo "$m"
# 							echo "$n"
# 							echo '$t'
# 							echo "$p"
# 							sbatch Submit_slurm_job_training_Atari_additionalGames.sh true $s $m $n $r $t $p ${Game}
# 						done
# 					done
# 				done
# 			done
# 		done 
# 	done
	
# done







echo "job submission done"















#echo "stage 3 training with different with different quantization settings"



# declare -a CodeBookSizeS=(4 16 64 512)
# declare -a Quantization_methodS=("VQVAE")
# declare -a n_segments=(1 4 16 64)
# declare -a seedS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# declare -a Quantization_Targets=("edge_sum")
# declare -a RatioDataForTraining=(1.0)






############2D shapes


# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_2D.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_2D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done




# #3D Cube shapes

# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_3D.sh false 0 None 0 $r None $p
# 	done
# done

# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do	
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_3D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done








# ##3-Body physics 

# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 			sbatch Submit_slurm_job_training_balls.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_balls.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done


# ##########Atari Pong
# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_AtariPong.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_AtariPong.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done





# #####################Space Invader
# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_SpaceInvader.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_SpaceInvader.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done



####################additional atari game


# declare -a Games=("BankHeist" "Berzerk" "Pong" "SpaceInvaders")

# #declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")

# for Game in "${Games[@]}"
# do 

# 	for r in "${seedS[@]}"
# 	do
# 		for p in "${RatioDataForTraining[@]}"
# 		do
# 			sbatch Submit_slurm_job_training_AtariGames.sh false 0 None 0 $r None $p ${Game}
# 		done
# 	done


# 	for r in "${seedS[@]}"
# 	do
# 		for s in "${CodeBookSizeS[@]}" 
# 		do		
# 			for m in "${Quantization_methodS[@]}"
# 			do

# 				for n in "${n_segments[@]}"
# 				do 
# 					for t in "${Quantization_Targets[@]}"
# 					do
# 						for p in "${RatioDataForTraining[@]}"
# 						do
# 							echo "$r"
# 							echo "$s"
# 							echo "$m"
# 							echo "$n"
# 							echo "$t"
# 							echo "$p"
# 							sbatch Submit_slurm_job_training_AtariGames.sh true $s $m $n $r $t $p ${Game}
# 						done
# 					done
# 				done
# 			done
# 		done 
# 	done
# done

# echo "job submission done"








# echo "stage 4, compare gumbel vs. VAQVAE"


# declare -a CodeBookSizeS=(64 512)
# declare -a Quantization_methodS=("gumbel")
# declare -a n_segments=(1)
# declare -a seedS=(1 2 3 4 5)
# declare -a Quantization_Targets=("edge_sum")
# declare -a RatioDataForTraining=(1.0)


# ###########2D shapes


# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_2D.sh false 0 None 0 $r None $p
# 	done
# done


# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_2D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done




# #3D Cube shapes

# for r in "${seedS[@]}"
# do
# 	for p in "${RatioDataForTraining[@]}"
# 	do
# 		sbatch Submit_slurm_job_training_3D.sh false 0 None 0 $r None $p
# 	done
# done

# for r in "${seedS[@]}"
# do
# 	for s in "${CodeBookSizeS[@]}" 
# 	do		
# 		for m in "${Quantization_methodS[@]}"
# 		do

# 			for n in "${n_segments[@]}"
# 			do 
# 				for t in "${Quantization_Targets[@]}"
# 				do	
# 					for p in "${RatioDataForTraining[@]}"
# 					do
# 						echo "$r"
# 						echo "$s"
# 						echo "$m"
# 						echo "$n"
# 						echo '$t'
# 						echo "$p"
# 						sbatch Submit_slurm_job_training_3D.sh true $s $m $n $r $t $p
# 					done
# 				done
# 			done
# 		done
# 	done 
# done


