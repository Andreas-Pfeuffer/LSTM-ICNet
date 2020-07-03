

import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

#import datasets
from cityscapes import *
from kitti import *
from virtual_kitti import *

# -----------------------------------
# pathes to change
# -----------------------------------
# 
# data_dir: path to datasets
# 
default_data_dir = '/media/festplatte_4T/Bilder/Benchmarks'
#
# -----------------------------------

default_model = 'ICNet_BN'
default_dataset = 'cityscapes_sequence_4_color_19'
default_architecture = 'semantic_segmentation'
default_weather='all_train'

def init_config(model=default_model, dataset=default_dataset, weather=default_weather, architecture=default_architecture, dataset_scale=1.0):
    
    # init config:

    config = edict()

    # general settings

    config.ENABLE_VISUAL_MONITORING = False
    config.ENABLE_TRAINING_PROTOCOL = True

    config.ARCHITECTURE = architecture     # Type 1: 'semantic_segmentation'
                                           # Type 2: 'object_detection'

    config.DATA_DIR = default_data_dir 




    # model

    config.MODEL = model
    config.MODEL_VARIANT = None  # variant of chosen model (only supported by special models. For others, set to None)
    
    config.USE_PRETRAINED_MODEL = True
    config.PRETRAINED_MODEL = 'pretrained_models/icnet_cityscapes_train_30k_bnnomerge' # cityscapes


    config.MODEL_NAME = 'model.ckpt'  # name of the model



    # define training set information

    config.SNAPSHOT_DIR = 'snapshots/'


    # training parameters

    config.MAX_ITERATIONS = 200
    config.BATCH_SIZE = 1 #16   
    config.ITER_SIZE = 1
    config.TIMESEQUENCE_LENGTH = 1
    config.NUM_STEPS = 180
    config.SAVE_PRED_EVERY = 100 #00
    

    config.UPDATE_MEAN_VAR = True    # whether to get update_op from tf.Graphic_Keys
    config.TRAIN_BETA_GAMMA = True   # whether to train beta & gamma in bn layer
    config.FILTER_SCALE = 1 # help="1 for using pruned model, while 2 for using non-pruned model."
    config.PRINT_TRAINABLE_PARAMETERS = False # whether to print all prameters, which are trained

    config.USE_GRADIENT_CLIPPING = True # whether to use gradient clipping
    config.MAX_GRAD_NORM = 10.0 # max norm for gradient scaling
    
    
    # training optimizer

    config.OPTIMIZER = edict()

    config.OPTIMIZER.TYPE = 'MomentumOptimizer'  # Type 1: MomentumOptimizer
                                                  # Type 2: AdamOptimizer
                                                  # Type 3: GradientDescentOptimizer
    config.OPTIMIZER.LEARNING_RATE = 1e-3
    config.OPTIMIZER.LEARNING_RATE_POLICY = 'poly' # Type 1: 'poly': poly learning rate policy
                                                   # Type 2: 'step': step learning rate policy: reduce learning rate by STEP_DECAY after STEP_SIZE iterations
    config.OPTIMIZER.MOMENTUM = 0.9
    config.OPTIMIZER.POWER = 0.9
    config.OPTIMIZER.WEIGHT_DECAY = 0.0001 
    config.OPTIMIZER.BIAS_DECAY = False           # Whether to have weight decay on bias as well # NOTE: only supported for object-detection until now!
    config.OPTIMIZER.STEP_SIZE = 35000
    config.OPTIMIZER.STEP_DECAY = 0.1


    # loss parameters

    config.LOSS = edict()

    config.LOSS.TYPE = 'softmax_cross_entropy'   # Type 1: softmax_cross_entropy

    # define training loss parameters (# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss)

    config.LOSS.LAMBDA1 = 0.16 #0.16 # 0.4
    config.LOSS.LAMBDA2 = 0.4 #0.4 # 0.6
    config.LOSS.LAMBDA3 = 1.0

    # other stuff

    config.MEASURE_TIME = False # help="whether to measure inference time"

    # freeze inference graph

    config.FREEZEINFERENCEGRAPH = edict()
    
    config.FREEZEINFERENCEGRAPH.MODE = False                     # if (True): inference graph will be freezed
    config.FREEZEINFERENCEGRAPH.PRINT_OUTPUT_NODE_NAMES = True     # print output node names (true/false)
    # string, which contains the output node names. Each output node name is separated by a comma
    config.FREEZEINFERENCEGRAPH.INPUT_NODE_NAMES = "image_batch"        
                                                                    # example: for ICNet_BN: input_node_names = "image_batch"
    config.FREEZEINFERENCEGRAPH.OUTPUT_NODE_NAMES = "ArgMax"        
                                                                    # example: for ICNet_BN: output_node_names = "ArgMax"


    # inference parameters

    config.INFERENCE = edict()

    config.INFERENCE.SAVEDIR_IMAGES = 'output/'
    config.INFERENCE.OVERLAPPING_IMAGE = 0.5
    config.INFERENCE.READ_IMAGE_FROM_LIST = False
    config.INFERENCE.DATA_LIST_PATH = ''

    config.INFERENCE.MODELPATH = config.SNAPSHOT_DIR

    # evaluation parameters

    config.EVALUATION = edict()

    config.EVALUATION.MODELPATH = config.SNAPSHOT_DIR
    config.EVALUATION.DOEVALUATION = True # enable evaluation during training
    config.EVALUATION.BEFORETRAINING = False # evaluate pretraiend network before starting with the training 
    config.EVALUATION.DOEVALUATION_TRAIN = False # enable evaluation during training on training set

    # image reader parameters

    config.IMAGEREADER = edict()
    config.IMAGEREADER.TRAIN = edict()
    config.IMAGEREADER.VAL = edict()

    if config.ARCHITECTURE == 'semantic_segmentation':
        config.IMAGEREADER.TRAIN.SHUFFLE = True    # Wheater to shuffle image batches during training
        config.IMAGEREADER.TRAIN.FLIPPING = True   # Whether to randomly flip the inputs during the training
        config.IMAGEREADER.TRAIN.SCALING = True    # Whether to randomly scale the inputs during the training
        config.IMAGEREADER.TRAIN.SCALING_MIN = 0.5 # min scaling of data
        config.IMAGEREADER.TRAIN.SCALING_MAX = 2.0 # max scaling of data
        config.IMAGEREADER.TRAIN.KEEP_RATIO = True #True # Whether to keep image ratio (or to resize image to default Image Size)
        config.IMAGEREADER.TRAIN.KEEP_IMAGE_SIZE = False  # if true, the image is not scaled
        config.IMAGEREADER.TRAIN.SCALE2MINSIZE_mode = False # scale data, such that each data size is larger than SCALE2MINSIZE_minSize
        config.IMAGEREADER.TRAIN.GT_MODE = config.ARCHITECTURE 
        config.IMAGEREADER.TRAIN.DATA_AUGMENTATION_METHOD = "None" # SLM: Standard Learning Method --> no data augmentation
                                                                  # RLM: Robust Learning Method --> fit white polygons to camera and lidar data
                                                                  # RLMvideo: Robust Learning Method applied on video sequences --> fit white polygons to some frames of the video sequence
                                                                  # RLMvideoNoise: Robust Learning Method applied on video sequences --> disturbe some frames of the video sequence by random noise
                                                                  # RLM2: Robust Learning Method 2 --> advanced RLM
                                                                  # RLMvideo2: Robust Learning Method 2 applied on video sequences --> use different augmentation methods 
                                                                  # RLMvideo3: Robust Learning Method 3 applied on video sequences --> use different augmentation methods 
                                                                  # rain: simulated rainy images  

        config.IMAGEREADER.VAL.SHUFFLE = False    # Wheater to shuffle image batches during training
        config.IMAGEREADER.VAL.FLIPPING = False   # Whether to randomly flip the inputs during the training
        config.IMAGEREADER.VAL.SCALING = False    # Whether to randomly scale the inputs during the training
        config.IMAGEREADER.VAL.SCALING_MIN = 1.0  # min scaling of data
        config.IMAGEREADER.VAL.SCALING_MAX = 1.0  # max scaling of data
        config.IMAGEREADER.VAL.KEEP_RATIO = True #True  # Whether to keep image ratio (or to resize image to default Image Size)
        config.IMAGEREADER.VAL.KEEP_IMAGE_SIZE = False  # if true, the image is not scaled
        config.IMAGEREADER.VAL.SCALE2MINSIZE_mode = False # scale data, such that each data size is larger than SCALE2MINSIZE_minSize
        config.IMAGEREADER.VAL.GT_MODE = config.ARCHITECTURE 
        config.IMAGEREADER.VAL.DATA_AUGMENTATION_METHOD = "None"
    else:
        print ('[ERROR] network architecture does not exist!!! Please check your spelling!')
        raise NotImplementedError

    

    # use dataset

    config.DATASET_NAME = dataset
    config.DATASET_WEATHER = weather
    config.DATASET_TRAIN = edict()
    config.DATASET_VAL = edict()
    config.DATASET_TEST = edict()
    config.DATASET_ALL = edict()

    config = init_dataset(config, dataset_scale=dataset_scale)

    # for time sequences

    config.USAGE_TIMESEQUENCES = config.DATASET_TRAIN.USAGE_TIMESEQUENCES # set to true for time sequeneces
    config.TIMESEQUENCES_SLIDINGWINDOW = config.DATASET_TRAIN.TIMESEQUENCES_SLIDINGWINDOW  # True: for each image of the time sequence, the groundtruth is known
                                                # False: the ground-truth is only known for the last image of the sequence
    config.TIMESEQUENCE_LENGTH = config.DATASET_TRAIN.TIMESEQUENCE_LENGTH

    return config



def init_dataset(config, dataset_scale=1.0):



    # ---------------------------
    # Kitti
    # ---------------------------

    # semantic segmentation

    if config.DATASET_NAME == 'kitti':
        config.DATASET_TRAIN = init_config_kitti(config.DATA_DIR, 'train')
        config.DATASET_VAL = init_config_kitti(config.DATA_DIR, 'val')

    # -----------------------------
    # Cityscapes
    # -----------------------------

    # semantic segmentation

    elif config.DATASET_NAME == 'cityscapes_color_19':
        config.DATASET_TRAIN = init_config_cityscapes_color_19(config.DATA_DIR, 'train', config.MODEL, config.DATASET_WEATHER, dataset_scale=dataset_scale)
        config.DATASET_VAL = init_config_cityscapes_color_19(config.DATA_DIR, 'val', config.MODEL, config.DATASET_WEATHER, dataset_scale=dataset_scale)
    elif config.DATASET_NAME =='cityscapes_sequence_4_color_19':
        config.DATASET_TRAIN = init_config_cityscapes_sequence_4_color_19(config.DATA_DIR, 'train', config.MODEL, config.DATASET_WEATHER, dataset_scale=dataset_scale)
        config.DATASET_VAL = init_config_cityscapes_sequence_4_color_19(config.DATA_DIR, 'val', config.MODEL, config.DATASET_WEATHER, dataset_scale=dataset_scale)

    # ----------------------------------
    # virtual Kitti
    # ----------------------------------

    # semantic segmentation

    elif config.DATASET_NAME == 'virtualKitti_color_14':
        config.DATASET_TRAIN = init_config_virtualKitti_color_14(config.DATA_DIR, 'train', config.MODEL)
        config.DATASET_VAL = init_config_virtualKitti_color_14(config.DATA_DIR, 'val', config.MODEL)
    elif config.DATASET_NAME =='virtualKitti_sequence_4_color_14':
        config.DATASET_TRAIN = init_config_virtualKitti_sequence_4_color_14(config.DATA_DIR, 'train', config.MODEL)
        config.DATASET_VAL = init_config_virtualKitti_sequence_4_color_14(config.DATA_DIR, 'val', config.MODEL)

    # -----------------------------------
    # else/unkown dataset
    # -----------------------------------

    else:
        print ('[ERROR] Dataset does not exist! Please check your spelling!')
        raise NotImplementedError


    # -----------------------------------
    # update config according to dataset
    # -----------------------------------

    # update image size

    

    return config



















