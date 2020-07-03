import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict



def init_config_kitti(dataDir, dataset):
    ''' choose dataset: available datasets: train, val'''


    config = edict()

    # training data

    if dataset == 'train':
        config.PATH2DATALIST = dataDir + '/' + '/KITTI/data_semantics/training_train1.txt'
    elif dataset == 'val':
        config.PATH2DATALIST = dataDir + '/' + '/KITTI/data_semantics/training_val1.txt'
    else:
        print ("[init_config_kitti] unknown dataset. Please choose 'train' or 'val' as argument!")

    # time sequences:

    config.USAGE_TIMESEQUENCES = False
    config.TIMESEQUENCES_SLIDINGWINDOW = False
    config.TIMESEQUENCE_LENGTH = 1 # length of a time sequence

    # define image properties and dataset properties

    config.INPUT_SIZE = [376, 1242]  # Input size for Kitti-images 
    config.USE_IMAGE_ROI = False # use only a part of the input image (true/false) 
    if config.USE_IMAGE_ROI:
        config.IMAGE_ROI = edict()
        config.IMAGE_ROI_MIN_X = 0 
        config.IMAGE_ROI_MAX_X = 376
        config.IMAGE_ROI_MIN_Y = 0
        config.IMAGE_ROI_MAX_Y = 1242 
        # update INPUT_SIZE
        config.USE_IMAGE_ROI = True
        config.INPUT_SIZE = [config.IMAGE_ROI_MAX_X-config.IMAGE_ROI_MIN_X, config.IMAGE_ROI_MAX_Y-config.IMAGE_ROI_MIN_Y]
    config.NUM_CHANNELS = 3  # 1 = grayscale image, 3 = color image
    config.IMG_MEAN = np.array((97.831, 101.6070, 96.673), dtype=np.float32)  # kitti segemnation train
    config.LOSS_CLASSWEIGHT = np.ones(12)

    config.NUM_CLASSES = 12 
    config.CLASSES = ['Unlabled', 'Road', 'Sidewalk', 'Pole', 'Traffic_Light', 'Traffic_Sign', 'Pedestrian', 'Bicyclist', 'Car', 'Truck', 'Bus', 'Motorcycle']
    config.IGNORE_LABEL = 255 # The class number of background

    return config








