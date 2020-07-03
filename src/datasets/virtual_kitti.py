import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict




def init_config_virtualKitti_color_14(dataDir, dataset, model):

    config = edict()

    # training data

    if dataset == 'train':
        config.PATH2DATALIST = dataDir + '/' + '/virtuellKitti/TrainingDir/train.txt'
    elif dataset == 'val':
        config.PATH2DATALIST = dataDir + '/' + '/virtuellKitti/TrainingDir/val.txt'
    else:
        print ("[virtualKitti_color_14] unknown dataset. Please choose 'train' or 'val' as argument!")

    # time sequences:

    config.USAGE_TIMESEQUENCES = False
    config.TIMESEQUENCES_SLIDINGWINDOW = False
    config.TIMESEQUENCE_LENGTH = 1 # length of a time sequence

    # define image properties and dataset properties

    config.INPUT_SIZE = [375, 1242]  # Input size for Kitti-images 
    config.USE_IMAGE_ROI = False # use only a part of the input image (true/false) 
    if config.USE_IMAGE_ROI:
        config.IMAGE_ROI = edict()
        config.IMAGE_ROI_MIN_X = 0 
        config.IMAGE_ROI_MAX_X = 1024
        config.IMAGE_ROI_MIN_Y = 0
        config.IMAGE_ROI_MAX_Y = 2048 
        # update INPUT_SIZE
        config.USE_IMAGE_ROI = True
        config.INPUT_SIZE = [config.IMAGE_ROI_MAX_X-config.IMAGE_ROI_MIN_X, config.IMAGE_ROI_MAX_Y-config.IMAGE_ROI_MIN_Y]

    config.NUM_CHANNELS = 3  # 1 = grayscale image, 3 = color image
    config.IMG_MEAN = np.array((92.18326241, 104.01986596,  95.49157379), dtype=np.float32)
    config.LOSS_CLASSWEIGHT = np.array([0.0237, 0.145, 0.0388, 1.3414, 1., 0.7167, 4.1887, 1.5826, 0.0555, 0.7596, 0.2211, 0.7369, 6.542 , 0.1283, 3.3077, 3.7557, 3.7825, 9.1408, 3.0635])

    config.NUM_CLASSES = 14 
    config.CLASSES = ['Misc','Building', 'Car', 'GuardRail', 'Pole', 'Road', 'Sky', 'Terrain', 'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']
    config.IGNORE_LABEL = 255 # The class number of background

    return config



def init_config_virtualKitti_sequence_4_color_14(dataDir, dataset, model):

    config = edict()

    # training data

    if dataset == 'train':
        config.PATH2DATALIST = dataDir + '/' + '/virtuellKitti/TrainingDir/train_sequence_4.txt'
    elif dataset == 'val':
        config.PATH2DATALIST = dataDir + '/' + '/virtuellKitti/TrainingDir/val_sequence_4.txt'
    else:
        print ("[virtualKitti_sequence_4_color_14] unknown dataset. Please choose 'train' or 'val' as argument!")

    # time sequences parameters 

    config.USAGE_TIMESEQUENCES = True
    config.TIMESEQUENCES_SLIDINGWINDOW = False
    config.TIMESEQUENCE_LENGTH = 4 # length of a time sequence

    # define image properties and dataset properties

    config.INPUT_SIZE = [375, 1242]  # Input size for Kitti-images 
    config.USE_IMAGE_ROI = False # use only a part of the input image (true/false) 
    if config.USE_IMAGE_ROI:
        config.IMAGE_ROI = edict()
        config.IMAGE_ROI_MIN_X = 0 
        config.IMAGE_ROI_MAX_X = 1024
        config.IMAGE_ROI_MIN_Y = 0
        config.IMAGE_ROI_MAX_Y = 2048 
        # update INPUT_SIZE
        config.USE_IMAGE_ROI = True
        config.INPUT_SIZE = [config.IMAGE_ROI_MAX_X-config.IMAGE_ROI_MIN_X, config.IMAGE_ROI_MAX_Y-config.IMAGE_ROI_MIN_Y]

    config.NUM_CHANNELS = 3  # 1 = grayscale image, 3 = color image
    config.IMG_MEAN = np.array((92.18326241, 104.01986596,  95.49157379), dtype=np.float32)
    config.LOSS_CLASSWEIGHT = np.array([0.0237, 0.145, 0.0388, 1.3414, 1., 0.7167, 4.1887, 1.5826, 0.0555, 0.7596, 0.2211, 0.7369, 6.542 , 0.1283, 3.3077, 3.7557, 3.7825, 9.1408, 3.0635])

    config.NUM_CLASSES = 14 
    config.CLASSES = ['Misc','Building', 'Car', 'GuardRail', 'Pole', 'Road', 'Sky', 'Terrain', 'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']
    config.IGNORE_LABEL = 255 # The class number of background

    return config


