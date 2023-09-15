#!/bin/bash
# This code runs in CPU, no GPUs are necessary
# Dataset parameters #####################################
DATASET=both
TISSUE=all
ALL_VS_ONE=False
MEAN_THR=-10.0
STD_THR=0.01
RAND_FRAC=1.0
GENE_LIST_CSV=None
BATCH_NORM=normal
FOLD_NUMBER=5
SEED=0
# Experiment name #########################################
EXP_NAME=automatic


# Experiments in toil/wang/recount3 with sample_frac=0.99
python sota_detection.py --source toil     --sample_frac 0.99  --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED
python sota_detection.py --source wang     --sample_frac 0.99  --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED
python sota_detection.py --source recount3 --sample_frac 0.99  --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED

# Experiments in toil/wang/recount3 with sample_frac=0.5
python sota_detection.py --source toil     --sample_frac 0.5   --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED
python sota_detection.py --source wang     --sample_frac 0.5   --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED
python sota_detection.py --source recount3 --sample_frac 0.5   --exp_name $EXP_NAME --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED