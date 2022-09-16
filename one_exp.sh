#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
DATASET=both
TISSUE=all
ALL_VS_ONE=False
BATCH_NORM=normal
# Training parameters #####################################
LR=0.00001
BATCH_SIZE=100
EPOCHS=20
TRAIN_SAMPLES=-1
# Paths ###################################################
EXP_NAME=both_normal_norm_weights
# Mode ####################################################
MODE=train # train or test

CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --batch_norm $BATCH_NORM --lr $LR --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --train_samples $TRAIN_SAMPLES --exp_name $EXP_NAME





