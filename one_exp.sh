#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
DATASET=both
TISSUE=all
ALL_VS_ONE=False
MEAN_THR=-10.0
STD_THR=0.0
BATCH_NORM=normal
# Training parameters #####################################
LR=0.00001
WEIGHTS=True
BATCH_SIZE=100
EPOCHS=20
# Paths ###################################################
EXP_NAME=CanDLE_classification
# Mode ####################################################
MODE=both # train, test or both
# Run main 
CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --batch_norm $BATCH_NORM --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME