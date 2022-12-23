#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
TISSUE=all
ALL_VS_ONE=False
MEAN_THR=-10.0
STD_THR=0.01
RAND_FRAC=1.0
SAMPLE_FRAC=0.99
GENE_LIST_CSV=None
BATCH_NORM=normal
FOLD_NUMBER=5
SEED=0
# Training parameters #####################################
LR=0.00001
WEIGHTS=True
BATCH_SIZE=100
EPOCHS=20
# Mode ####################################################
MODE=train # train, test or both


# Run toil/wang/recount3 classification results with both datasets
DATASET=both
EXP_NAME=CanDLE_classification_both_toil_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_both_wang_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_both_recount3_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

# Run toil/wang/recount3 classification results with tcga dataset
DATASET=tcga
EXP_NAME=CanDLE_classification_tcga_toil_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_tcga_wang_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_tcga_recount3_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

# Run toil/wang/recount3 classification results with gtex dataset
DATASET=gtex
EXP_NAME=CanDLE_classification_gtex_toil_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_gtex_wang_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_gtex_recount3_sample_frac_0.99
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --fold_number $FOLD_NUMBER --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

