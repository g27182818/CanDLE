#!/bin/bash
# Dataset parameters #####################################
SOURCE=recount3
DATASET=both #      Do not change
TISSUE=all #        Do not change
ALL_VS_ONE=False #  Do not change
MEAN_THR=-10.0
STD_THR=0.01
RAND_FRAC=1.0
SAMPLE_FRAC=0.5
GENE_LIST_CSV=None
BATCH_NORM=normal
FOLD_NUMBER=5
SEED=0 #            Do not change


# Run bias check code 
python bias_check.py --source $SOURCE --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED
