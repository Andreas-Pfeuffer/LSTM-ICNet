#!/bin/bash


MODEL="Faster_LSTM_ICNet_v5"
MODEL_VARIANT="depthwise_end2end_half"
ARCHTIECTURE="semantic_segmentation"
DATASET_1="cityscapes_sequence_4_color_19"   
BATCH_SIZE="1"
RESULT_DIR="results/2020_02_14b_Faster_LSTM_ICNet_v5_cityscape_sequence_4_color_19_batch1_60k"


# evalate model

singularity exec --nv -B /media/ singularity/ubuntu1804_tensorflow1.14_cuda10.simg python inference.py --architecture=$ARCHTIECTURE --model=$MODEL --dataset=$DATASET_1 --pretrained-model=$RESULT_DIR --evaluation_set=val --weather=all_train --model-variant=$MODEL_VARIANT
