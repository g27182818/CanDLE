#!/bin/bash
GPU=1
DATASET=tcga
MODE=compute

# Initiall setup of experimental factors
WEIGHTS=True
SAMPLE_FRAC=0.5

python all_vs_one_exp.py --source toil      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source wang      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source recount3  --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE

# Remove weights
WEIGHTS=False
SAMPLE_FRAC=0.5

python all_vs_one_exp.py --source toil      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source wang      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source recount3  --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE

# Remove sample fraction requirement
WEIGHTS=True
SAMPLE_FRAC=0.0

python all_vs_one_exp.py --source toil      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source wang      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source recount3  --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE

# No sample fraction and no weights
WEIGHTS=False
SAMPLE_FRAC=0.0

python all_vs_one_exp.py --source toil      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source wang      --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE
python all_vs_one_exp.py --source recount3  --weights $WEIGHTS --sample_frac $SAMPLE_FRAC --gpu $GPU --dataset $DATASET --mode $MODE




