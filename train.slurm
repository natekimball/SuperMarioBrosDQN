#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100
# #SBATCH --reservation=bi_fox_dgx
#SBATCH --job-name=rl-training
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem=256G

date
nvidia-smi
module purge
module load singularity

echo "Starting training"
time singularity exec --nv images/gym.sif python mario.py --time 90000
record=$(tail -n 1 temp)
echo "PROGRESS #1 RECORD: $record"

for ((i=2; i<=11; i++))
do
    time singularity exec --nv images/gym.sif python mario.py --time 90000 --load_model models/mario --no_epsilon --record $record
    record=$(tail -n 1 temp)
    echo "PROGRESS #$i RECORD: $record"
done

echo "Training complete"