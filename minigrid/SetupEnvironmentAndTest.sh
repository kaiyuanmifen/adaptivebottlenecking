
#onlocal computer
cd /mnt/c/Users/kaiyu/Google\ Drive/research/MILA/RIM/
source MiniGrid_RIM_Env_New_python3.7/bin/activate



#on Hcluster
#srun --pty -p interactive --mem 20G -t 0-06:00 /bin/bash
srun -n 1 --pty -t 2:00:00 --mem 15G -p gpu --gres=gpu:1 bash
module load gcc/6.2.0 cuda/10.1
module load python/3.7.4
#module load conda2/4.2.13
source /home/dl249/RIM/MiniGrid_RIM_Env_New/bin/activate

virtualenv -p python3 MiniGrid_RIM_Env_New --system-site-packages



pip install -r reuqirements.txt

pip install torch torchvision

pip install numpy==1.18.0

pip install tqdm

pip install torch_ac>=1.1.0

pip install tensorboardX>=1.6




python3 evaluate.py --env  MiniGrid-DLTestEnvS10N3AllGoal-random-v0 --model DLEnvs_RIMSP_trial --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults False --TaskName Testing

python3 train.py --algo ppo --env MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP_trial --frames 500000 --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4


python3 visualize.py --env MiniGrid-MultiRoom-N6-S4-v0 --model DLEnvs_RIMSP_trial --seed 8 --gif DLEnvs_RIMSP8 --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 


python3 train.py --algo ppo --env MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP_trial2 --frames 50000 --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4
python3 train.py --algo ppo --env MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP --frames 10000 --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --model_dir_save "storage/Testing"



TrainingEnv="DLEnvs"
TestingEnv="MiniGrid-DLEnv-random-v0"


python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_LSTM" --seed 10113 --gif "${TrainingEnv}_${TestingEnv}_LSTM" --episodes 1
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIM" --seed 120014 --gif "${TrainingEnv}_${TestingEnv}_RIM"  --use_rim --rim_type Original --NUM_UNITS 4 --k 3 --episodes 1
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 1565 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1


TrainingEnv="DLEnvs"
TestingEnv='MiniGrid-DLSWEnvW7H9T1Objects-random-v0'

python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_LSTM" --seed 10112 --gif "${TrainingEnv}_${TestingEnv}_LSTM" --episodes 1
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIM" --seed 12001 --gif "${TrainingEnv}_${TestingEnv}_RIM"  --use_rim --rim_type Original --NUM_UNITS 4 --k 3 --episodes 1
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 1562 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1


python3 visualize.py --env "MiniGrid-DLEnv-random-v0" --model "DLEnvs_RIMSP" --seed 1562 --gif "Tiral_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1
python3 visualize.py --env 'MiniGrid-DLSWEnvW7H9T1Objects-random-v0' --model "DLEnvs_RIMSP" --seed 1562 --gif "Tira2_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1


TrainingEnv="DLEnvs"
TestingEnv='MiniGrid-DLSWEnvW7H9T1Objects-random-v0'
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 15633 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1


TrainingEnv="DLEnvs"
TestingEnv="MiniGrid-DLEnv-random-v0"
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 1768 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1



TrainingEnv="DLEnvs"
TestingEnv='MiniGrid-DLSWEnvW5H5T0Goal-random-v0'
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 15653 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1


TrainingEnv="DLEnvs"
TestingEnv='MiniGrid-DLTestingW16H16T2Objects-random-v0'
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 15653 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1



TrainingEnv="DLEnvs"
TestingEnv='MiniGrid-DLTestingW9H9T1Objects-random-v0'
python3 visualize.py --env "$TestingEnv" --model "${TrainingEnv}_RIMSP" --seed 15611 --gif "${TrainingEnv}_${TestingEnv}_RIMSP" --use_rim --rim_type Shared --NUM_UNITS 4 --k 3 --Number_active 2 --num_schemas 4 --episodes 1



python3 train.py --algo ppo --env MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP --frames 10000 --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --model_dir_save "storage/Testing" --freeze_sharedParameters


./train_schema_weighting.sh 'MiniGrid-DLSWEnvW9H7T1Goal-random-v0' 1510000


python3 SchemaWeightingPredictor.py 

python3 evaluate.py --env  MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults False --TaskName "Miega" --PredictSchemaWeights --Width 5 --Height 10 --BlockType 2 --Targets "Objects"
python3 evaluate.py --env  MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults False --TaskName "Miega"

python3 evaluate.py --env  MiniGrid-DLEnv-random-v0 --model DLEnvs_RIMSP --use_rim --rim_type Shared --NUM_UNITS 4 --k 2 --Number_active 2 --num_schemas 4 --StoreResults False --TaskName "Miega" --PredictSchemaWeights --Width 5 --Height 10 --BlockType 2 --Targets "Objects"


./ExperimentalEvaluation.sh DLEnvs_RIMSP true Shared