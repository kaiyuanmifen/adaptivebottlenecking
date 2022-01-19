#!/bin/bash
#SBATCH --job-name=training
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=40G               # memory (per node)
#SBATCH --time=0-02:20            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate GNN




###Additional atari games


#declare -a Games=("Boxing")


# declare -a Games=("Boxing" "DoubleDunk" "MsPacman" "Alien" "BankHeist" "Berzerk" "Pong" "SpaceInvaders")

# for Game in "${Games[@]}"
# do
# 	echo ${Game}
# 	python data_gen/env.py --env_id ${Game}"Deterministic-v0" --fname "../../data/"${Game}"_train.h5" --num_episodes 10 --atari --seed 1
# 	python data_gen/env.py --env_id ${Game}"Deterministic-v0" --fname "../../data/"${Game}"_eval.h5" --num_episodes 10 --atari --seed 2
# 	#python data_gen/env.py --env_id ${Game}"NoFrameskip-v0" --fname "data/"${Game}"_eval_OOD.h5" --num_episodes 100 --atari --seed 3
# 	python data_gen/env.py --env_id ${Game}"NoFrameskip-v4" --fname "../../data/"${Game}"_eval_OOD2.h5" --num_episodes 10 --atari --seed 3

# done

###minigrid 
# Game="Minigrid"
# Env=MiniGrid-KeyCorridorS3R2-v0
# echo ${Game}
# python data_gen/env.py --env_id MiniGrid-KeyCorridorS3R2-v0 --fname "data/"${Game}"_train.h5" --num_episodes 1000 --atari --seed 1
# python data_gen/env.py --env_id MiniGrid-KeyCorridorS3R2-v0 --fname "data/"${Game}"_eval.h5" --num_episodes 100 --atari --seed 2
# python data_gen/env.py --env_id MiniGrid-KeyCorridorS3R3-v0 --fname"data/"${Game}"_eval_OOD.h5" --num_episodes 100 --atari --seed 3




# ##Boxing

# python data_gen/env.py --env_id BoxingDeterministic-v0 --fname data/Boxing_train.h5 --num_episodes 1000 --atari --seed 1
# python data_gen/env.py --env_id BoxingDeterministic-v0 --fname data/Boxing_eval.h5 --num_episodes 100 --atari --seed 2




# ##DoubleDunk
# python data_gen/env.py --env_id DoubleDunkDeterministic-v0 --fname data/DoubleDunk_train.h5 --num_episodes 1000 --atari --seed 1
# python data_gen/env.py --env_id DoubleDunkDeterministic-v0 --fname data/DoubleDunk_eval.h5 --num_episodes 100 --atari --seed 2
# python data_gen/env.py --env_id DoubleDunkNoFrameskip-v0 --fname data/DoubleDunk_eval_OOD.h5 --num_episodes 100 --atari --seed 2


# ##Atari Pong 


# python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
# python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2

#python data_gen/env.py --env_id PongNoFrameskip-v0 --fname data/pong_eval_OOD.h5 --num_episodes 100 --atari --seed 2


# ##spaceInvader

# python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_train.h5 --num_episodes 1000 --atari --seed 1
# python data_gen/env.py --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_eval.h5 --num_episodes 100 --atari --seed 2

#python data_gen/env.py --env_id SpaceInvadersNoFrameskip-v0 --fname data/spaceinvaders_eval_OOD.h5 --num_episodes 100 --atari --seed 2


# ##2D

# python data_gen/env.py --env_id ShapesTrain-v0 --fname ../../data/shapes_train.h5 --num_episodes 1000 --seed 1
# python data_gen/env.py --env_id ShapesEval-v0 --fname ../../data/shapes_eval.h5 --num_episodes 10000 --seed 2

# python data_gen/env.py --env_id ShapesN2-v0 --fname ../../data/shapesN2.h5 --num_episodes 10000 --seed 3
# python data_gen/env.py --env_id ShapesN3-v0 --fname ../../data/shapesN3.h5 --num_episodes 10000 --seed 4
# python data_gen/env.py --env_id ShapesN4-v0 --fname ../../data/shapesN4.h5 --num_episodes 10000 --seed 5
# python data_gen/env.py --env_id ShapesN6-v0 --fname ../../data/shapesN6.h5 --num_episodes 10000 --seed 6




# #3D

#python data_gen/env.py --env_id CubesTrain-v0 --fname data/cubes_train.h5 --num_episodes 1000 --seed 3
#python data_gen/env.py --env_id CubesEval-v0 --fname data/cubes_eval.h5 --num_episodes 10000 --seed 4

# python data_gen/env.py --env_id CubesN2-v0 --fname data/cubesN2.h5 --num_episodes 10000 --seed 6
# python data_gen/env.py --env_id CubesN3-v0 --fname data/cubesN3.h5 --num_episodes 10000 --seed 7
# python data_gen/env.py --env_id CubesN4-v0 --fname data/cubesN4.h5 --num_episodes 10000 --seed 8
# python data_gen/env.py --env_id CubesN6-v0 --fname data/cubesN6.h5 --num_episodes 10000 --seed 9




# #other atarti games

# declare -a AllTask=("Venture" "WizardOfWor" "Alien" "Tennis" "Phoenix" "Assault")

# for task in "${AllTask[@]}"
# do
# 	echo "working on "$task
# 	python data_gen/env.py --env_id ${task}"Deterministic-v4" --fname "data/"${task}"_train.h5" --num_episodes 1000 --atari --seed 1
# 	python data_gen/env.py --env_id ${task}"Deterministic-v4" --fname "data/"${task}"_eval.h5" --num_episodes 100 --atari --seed 2

# done 




###3 body physics 

python data_gen/physics.py --num-episodes 1000 --fname data/balls_train.h5 --seed 1
python data_gen/physics.py --num-episodes 1000 --fname data/balls_eval.h5 --eval --seed 2

python data_gen/physics.py --num-episodes 1000 --fname data/ballsR2.h5 --eval --seed 3 --radius 2

python data_gen/physics.py --num-episodes 1000 --fname data/ballsR1.h5 --eval --seed 4 --radius 1

python data_gen/physics.py --num-episodes 1000 --fname data/ballsR4.h5 --eval --seed 5 --radius 4


# python data_gen/physics.py --num-episodes 1000 --fname data/ballsR0.1.h5 --eval --seed 3 --radius 0.1

# python data_gen/physics.py --num-episodes 1000 --fname data/ballsR0.5.h5 --eval --seed 4 --radius 0.5

# python data_gen/physics.py --num-episodes 1000 --fname data/ballsR5.h5 --eval --seed 5 --radius 5