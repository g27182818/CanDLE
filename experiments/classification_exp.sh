#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
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
# Mode ####################################################
MODE=both # train, test or both


# Run toil/wang/recount3 classification results with both datasets
DATASET=both
EXP_NAME=CanDLE_classification_toil_both
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_wang_both
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_recount3_both
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

# Run toil/wang/recount3 classification results with tcga dataset
DATASET=tcga
EXP_NAME=CanDLE_classification_toil_tcga
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_wang_tcga
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_recount3_tcga
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

# Run toil/wang/recount3 classification results with gtex dataset
DATASET=gtex
EXP_NAME=CanDLE_classification_toil_gtex
CUDA_VISIBLE_DEVICES=$GPU python main.py --source toil      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_wang_gtex
CUDA_VISIBLE_DEVICES=$GPU python main.py --source wang      --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME
EXP_NAME=CanDLE_classification_recount3_gtex
CUDA_VISIBLE_DEVICES=$GPU python main.py --source recount3  --dataset $DATASET --tissue $TISSUE --all_vs_one $ALL_VS_ONE --mean_thr $MEAN_THR --std_thr $STD_THR --rand_frac $RAND_FRAC --sample_frac $SAMPLE_FRAC --gene_list_csv $GENE_LIST_CSV --batch_norm $BATCH_NORM --seed $SEED --lr $LR --weights $WEIGHTS --batch_size $BATCH_SIZE --epochs $EPOCHS --mode $MODE --exp_name $EXP_NAME

