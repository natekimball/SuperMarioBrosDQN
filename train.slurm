#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100
#SBATCH --reservation=bi_fox_dgx
#SBATCH --job-name=small-training
#SBATCH --output=small-%u-%j.out
#SBATCH --error=small-%u-%j.err
#SBATCH --mem=256G
# #SBATCH --exclusive

date
nvidia-smi
module purge
module load singularity

CHAPTERS=18
STEPS=50000
MODEL='small'
MODEL_DIR="models/$MODEL-new"
SCORES_FILE="$MODEL-model-scores-new.txt"

echo "Starting training $MODEL"

time singularity exec --nv images/gym.sif python mario.py --time=$STEPS --model=$MODEL --learning_rate=0.005 --save_model_dir=$MODEL_DIR >> $SCORES_FILE
record=$(tail -n 1 $SCORES_FILE)
echo "CHAPTER #1 COMPLETE, RECORD: $record"

for ((i=2; i<=$CHAPTERS; i++))
do
    if [ $i -lt 6 ]; then
        learning_rate=0.001
    else
        learning_rate=0.0005
    fi

    time singularity exec --nv images/gym.sif python mario.py --time=$STEPS --load_model=$MODEL_DIR --save_model_dir=$MODEL_DIR --no_epsilon --record=$record --learning_rate=$learning_rate >> $SCORES_FILE
    record=$(tail -n 1 $SCORES_FILE)
    echo "CHAPTER #$i COMPLETE, RECORD: $record"
done

echo "Training complete"