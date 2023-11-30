sync:
	-rsync -avz Makefile mario.py DQNAgent.py requirements.txt train.slurm tma5gv@rivanna:/scratch/tma5gv/SuperMarioRL
	
# rsync -ravz --exclude=*.sif --exclude=blob tma5gv@rivanna:/scratch/tma5gv/SuperMarioRL/images .

get-results:
	-rsync -ravz tma5gv@rivanna:/scratch/tma5gv/SuperMarioRL/out .

get-model:
	-rsync -ravz tma5gv@rivanna:/scratch/tma5gv/SuperMarioRL/models .

run:
	time singularity exec --bind `pwd`:/home --pwd /home --nv images/gym.sif python mario.py

# rsync -avz --exclude='MAR' ./ tma5gv@rivanna:/scratch/tma5gv/super-mario-RL


# source ~/anaconda/bin/activate
# conda activate MAR