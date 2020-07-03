import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict



def init_config_cityscapes_color_19(dataDir, dataset, model, weather='good', dataset_scale=1.0):

    config = edict()

    # training data

    if weather == 'all_train':
        if dataset == 'train':
            config.PATH2DATALIST = dataDir + '/' + '/Cityscape/Training/training_color_train.txt'
        elif dataset == 'val':
            config.PATH2DATALIST = dataDir + '/' + '/Cityscape/Training/training_color_val.txt'
        else:
            print ("[init_config_cityscapes_color_19] unknown dataset. Please choose 'train' or 'val' as argument!")
    elif weather == 'good':
        if dataset == 'train':
            config.PATH2DATALIST = dataDir + '/' + '/Cityscape/Training/training_color_train.txt'
        elif dataset == 'val':
            config.PATH2DATALIST = dataDir + '/' + '/Cityscape/Training/training_color_val.txt'
        else:
            print ("[init_config_cityscapes_color_19] unknown dataset. Please choose 'train' or 'val' as argument!")
    else:
        print ("[init_conifg_cityscapes_color_19] unknown weather conditions")

    # time sequences:

    config.USAGE_TIMESEQUENCES = False
    config.TIMESEQUENCES_SLIDINGWINDOW = False
    config.TIMESEQUENCE_LENGTH = 1 # length of a time sequence

    # define image properties and dataset properties

    config.INPUT_SIZE = [1024, 2048]  
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
    # apply dataset_scale
    config.INPUT_SIZE = [int(config.INPUT_SIZE[0] // dataset_scale), int(config.INPUT_SIZE[1] // dataset_scale)]
    print ('set input_size of dataset to:', config.INPUT_SIZE)

    config.NUM_CHANNELS = 3  # 1 = grayscale image, 3 = color image
    config.IMG_MEAN = np.array((72.392, 82.909, 73.158), dtype=np.float32)
    config.LOSS_CLASSWEIGHT = np.array([0.0237, 0.145, 0.0388, 1.3414, 1., 0.7167, 4.1887, 1.5826, 0.0555, 0.7596, 0.2211, 0.7369, 6.542 , 0.1283, 3.3077, 3.7557, 3.7825, 9.1408, 3.0635])

    config.NUM_CLASSES = 19 
    config.CLASSES = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic_Light', 'Traffic_Sign', 'Vegetation', 'Terrain',
                     'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train','Motorcycle', 'Bicycle']
    config.IGNORE_LABEL = 255 # The class number of background

    return config




def init_config_cityscapes_sequence_4_color_19(dataDir, dataset, model, weather='good', dataset_scale=1.0):
   
    config = edict()

    # training data

    if weather == 'good' or weather == 'all_train':
        if dataset == 'train':
            config.PATH2DATALIST = dataDir + '/' + 'Cityscape/Training/training_color_sequence_4_train.txt'
        elif dataset == 'val':
            config.PATH2DATALIST = dataDir + '/' + 'Cityscape/Training/training_color_sequence_4_val.txt'
        else:
            print ("[init_config_cityscapes_sequence_4_color_1] unknown dataset. Please choose 'train' or 'val' as argument!")
    else:
        print ("[init_config_cityscapes_sequence_4_color_1] unknown weather conditions")   
     
    

    # time sequences parameters 

    config.USAGE_TIMESEQUENCES = True
    config.TIMESEQUENCES_SLIDINGWINDOW = False
    config.TIMESEQUENCE_LENGTH = 4 # length of a time sequence

    # define image properties and dataset properties

    config.INPUT_SIZE = [1024, 2048]  
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
    # apply dataset_scale
    config.INPUT_SIZE = [int(config.INPUT_SIZE[0] // dataset_scale), int(config.INPUT_SIZE[1] // dataset_scale)]
    print ('set input_size of dataset to:', config.INPUT_SIZE)

    config.NUM_CHANNELS = 3  # 1 = grayscale image, 3 = color image
    config.IMG_MEAN = np.array((72.392, 82.909, 73.158), dtype=np.float32)
    config.LOSS_CLASSWEIGHT = np.ones(19)
    config.LOSS_CLASSWEIGHT = np.array([0.0237, 0.145, 0.0388, 1.3414, 1., 0.7167, 4.1887, 1.5826, 0.0555, 0.7596, 0.2211, 0.7369, 6.542 , 0.1283, 3.3077, 3.7557, 3.7825, 9.1408, 3.0635])

    config.NUM_CLASSES = 19 
    config.CLASSES = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic_Light', 'Traffic_Sign', 'Vegetation', 'Terrain',
                     'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train','Motorcycle', 'Bicycle']
    config.IGNORE_LABEL = 255 # The class number of background

    return config


