sync:
	rsync -avz Makefile mario.py DQNAgent.py requirements.txt train.slurm tma5gv@rivanna:/scratch/tma5gv/SuperMarioRL

run:
	time singularity exec --bind `pwd`:/home --pwd /home --nv images/gym.sif python mario.py

# rsync -avz --exclude='MAR' ./ tma5gv@rivanna:/scratch/tma5gv/super-mario-RL

# source ~/anaconda/bin/activate
# conda activate MAR