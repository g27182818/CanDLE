#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
SOURCE=recount3
DATASET=tcga
TISSUE=all
ALL_VS_ONE=False
MEAN_THR=-10.0
STD_THR=0.01
RAND_FRAC=1.0
SAMPLE_FRAC=0.5
GENE_LIST_CSV=None
BATCH_NORM=normal
SEED=0
# Training parameters #####################################
LR=0.00001
WEIGHTS=True
BATCH_SIZE=100
EPOCHS=20
# Names ###################################################
EXP_NAME=misc_test
# Mode ####################################################
MODE=both # train, test or both

# Run main 
CUDA_VISIBLE_DEVICES=$GPU python main.py --source $SOURCE --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

