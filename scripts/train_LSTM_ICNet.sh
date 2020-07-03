#!/bin/bash

EVAL="True" # True

MODEL="LSTM_ICNet_v5"
MODEL_VARIANT="end2end" 
ARCHTIECTURE="semantic_segmentation"
PRETRAINED_MODEL="pretrained_models/icnet_cityscapes_train_30k_bnnomerge"
ITERATIONS="60000"
BATCH_SIZE="1"
DATASET_1="cityscapes_sequence_4_color_19"  
DATA_AUGMENTATION="SLM"  # RLMvideo3
DATASET_SCALE="1"
OPTIMIZER="AdamOptimizer"
LEARNING_RATE="5e-5"
LOSS="softmax_cross_entropy"
RESULT_DIR="results/2020_07_01a_${MODEL}_${MODEL_VARIANT}_datasetScale${DATASET_SCALE}_${DATA_AUGMENTATION}_${DATASET_1}_batch${BATCH_SIZE}_gpu1_60k"

singularity exec --nv -B /media/ singularity/ubuntu1804_tensorflow1.14_cuda10.simg python train.py --architecture=$ARCHTIECTURE --pretrained-model=$PRETRAINED_MODEL --dataset=$DATASET_1 --result-dir=$RESULT_DIR --model="$MODEL"  --max-iterations=$ITERATIONS --batch-size=$BATCH_SIZE --data-augmentation=$DATA_AUGMENTATION  --optimizer=$OPTIMIZER --learning-rate=$LEARNING_RATE  --loss=$LOSS --dataset-scale=$DATASET_SCALE  --model-variant=$MODEL_VARIANT

if [ "$EVAL" = "True" ]
then
    # good weather conditions - training set
    singularity exec --nv -B /media/ singularity/ubuntu1804_tensorflow1.14_cuda10.simg python evaluate.py --architecture=$ARCHTIECTURE --model=$MODEL --dataset=$DATASET_1 --pretrained-model=$RESULT_DIR --evaluation_set=train --weather=all_train --model-variant=$MODEL_VARIANT

    # good weather conditions - validation set
    singularity exec --nv -B /media/ singularity/ubuntu1804_tensorflow1.14_cuda10.simg python evaluate.py --architecture=$ARCHTIECTURE --model=$MODEL --dataset=$DATASET_1 --pretrained-model=$RESULT_DIR --evaluation_set=val --weather=all_train --model-variant=$MODEL_VARIANT
else
    echo "Not evaluation"
fi
