#!/bin/bash
# GPU to use
GPU=1
# Dataset parameters #####################################
DATASET=both
TISSUE=all
ALL_VS_ONE=False
BATCH_NORM=normal
# Graph parameters ########################################
# Model parameters ########################################
MODEL=MLP_ALL
# Training parameters #####################################
LR=0.00001
BATCH_SIZE=100
EPOCHS=20
ADV_E_TEST=0.05
ADV_E_TRAIN=0.0
N_ITERS_APGD=50
NUM_TEST=69
TRAIN_SAMPLES=-1
# Paths ###################################################
EXP_NAME=both_normal_norm_weights
# Mode ####################################################
MODE=train # train or test

CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --batch_norm $BATCH_NORM --model $MODEL --lr $LR --batch_size $BATCH_SIZE --epochs $EPOCHS --adv_e_test $ADV_E_TEST --adv_e_train $ADV_E_TRAIN --n_iters_apgd $N_ITERS_APGD --mode $MODE --num_test $NUM_TEST --train_samples $TRAIN_SAMPLES --exp_name $EXP_NAME





