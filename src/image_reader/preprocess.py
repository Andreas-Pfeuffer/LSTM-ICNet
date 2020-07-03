
import os
import sys
import time

import numpy as np
import cv2
import random
from random import randint


from dataAugmentation_RLMvideo import dataAugmentation_RLMvideo
from dataAugmentation_RLMvideo3 import dataAugmentation_RLMvideo3



def preprocess(path2sample, config, config_Dataset):
    """ preprocess each sample"""


    data_list = []
    label_list = []


    # imread data and label
    
    for time_iter in xrange(len(path2sample)):
        data = cv2.imread(path2sample[time_iter][0], 3)
        if config_Dataset.USE_IMAGE_ROI:
            data = data[config_Dataset.IMAGE_ROI_MIN_X:config_Dataset.IMAGE_ROI_MAX_X, config_Dataset.IMAGE_ROI_MIN_Y:config_Dataset.IMAGE_ROI_MAX_Y,:]

        # get/load depth data
        if len(path2sample[time_iter]) == 3:
            if config_Dataset.NUM_CHANNELS == 4:
                depth = cv2.imread(path2sample[time_iter][1], 0)
                if config_Dataset.USE_IMAGE_ROI:
                    depth = depth[config_Dataset.IMAGE_ROI_MIN_X:config_Dataset.IMAGE_ROI_MAX_X, config_Dataset.IMAGE_ROI_MIN_Y:config_Dataset.IMAGE_ROI_MAX_Y]
            elif config_Dataset.NUM_CHANNELS == 6:
                depth = cv2.imread(path2sample[time_iter][1], 3)
                if config_Dataset.USE_IMAGE_ROI:
                    depth = depth[config_Dataset.IMAGE_ROI_MIN_X:config_Dataset.IMAGE_ROI_MAX_X, config_Dataset.IMAGE_ROI_MIN_Y:config_Dataset.IMAGE_ROI_MAX_Y,:]

        if config.GT_MODE == "semantic_segmentation":
            label = cv2.imread(path2sample[time_iter][-1], 0)
            if config_Dataset.USE_IMAGE_ROI:
                label = label[config_Dataset.IMAGE_ROI_MIN_X:config_Dataset.IMAGE_ROI_MAX_X, config_Dataset.IMAGE_ROI_MIN_Y:config_Dataset.IMAGE_ROI_MAX_Y]
        else:
            print ('[ERROR] network architecture does not exist!!! Please check your spelling!')
            raise NotImplementedError
        

        # ----------------------------------------------------
        # for Early Fusion: concatenate image and depth image
        # ----------------------------------------------------

        if len(path2sample[time_iter]) == 3:
            if config_Dataset.NUM_CHANNELS == 4:
                data = np.concatenate((data, np.expand_dims(depth, axis=2)), axis=2)
            elif config_Dataset.NUM_CHANNELS == 6:
                data = np.concatenate((data, depth), axis=2)

        data_list.append(data)
        label_list.append(label)


    # -------------------------------
    # data augmentation for videos
    # --------------------------------

    if config.DATA_AUGMENTATION_METHOD == "RLMvideo":
        data_list = dataAugmentation_RLMvideo(data_list)
    elif config.DATA_AUGMENTATION_METHOD == "RLMvideo3":
        data_list = dataAugmentation_RLMvideo3(data_list)

    

    # -------------------------------
    # substract image mean:
    # -------------------------------

    for time_iter in xrange(len(data_list)):
        data_list[time_iter] = substract_imageMean(data_list[time_iter], config_Dataset.IMG_MEAN)


    # -------------------------------
    # image flipping
    # -------------------------------

    if config.FLIPPING:
        if randint(0, 1):
            # flip data
            flip_data(data_list, label_list, config)

    # ------------------------------
    # scale image to blob size
    # ------------------------------

    if config.KEEP_RATIO: # keep image ratio
        data_shape = data_list[0].shape
        data_size_min = np.min(data_shape[0:2])
        data_size_max = np.max(data_shape[0:2])
        data_scale = float(config_Dataset.INPUT_SIZE[0]) / float(data_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE

        if np.round(data_scale * data_size_max) > config_Dataset.INPUT_SIZE[1]:
            data_scale = float(config_Dataset.INPUT_SIZE[1]) / float(data_size_max)

        data_scale_x = data_scale
        data_scale_y = data_scale
    elif config.KEEP_IMAGE_SIZE: # keep image size
        data_scale_x = 1
        data_scale_y = 1
    elif config.SCALE2MINSIZE_mode: # scale to minium data size
        im_shape = data_list[0].shape
        #print (im_shape)
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        data_scale = float(config.SCALE2MINSIZE_minSize) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(data_scale * im_size_max) > config.SCALE2MINSIZE_maxSize:
            im_scale = float(config.SCALE2MINSIZE_maxSize) / float(im_size_max)
        data_scale_x = data_scale
        data_scale_y = data_scale

    else: # resize image to input shape
        data_shape = data_list[0].shape
        data_scale_x = float(config_Dataset.INPUT_SIZE[1]) / float(data_shape[1])
        data_scale_y = float(config_Dataset.INPUT_SIZE[0]) / float(data_shape[0])

    # iterate here, such that all data of one time-sequence have identical blob_size
    for time_iter in xrange(len(path2sample)):
        data_list[time_iter] = cv2.resize(data_list[time_iter], None, None, fx=data_scale_x, fy=data_scale_y, interpolation=cv2.INTER_LINEAR)

        
        label_list[time_iter] = cv2.resize(label_list[time_iter], None, None, fx=data_scale_x, fy=data_scale_y, interpolation=cv2.INTER_LINEAR)
        label_list[time_iter] = np.expand_dims(label_list[time_iter], axis=2)


    # ------------------------------
    # random scaling
    # ------------------------------

    if config.SCALING:
        scale = np.random.uniform(config.SCALING_MIN, config.SCALING_MAX,1)
        data_scale_y *= scale
        data_scale_x *= scale

        # iterate here, such that all data of one time-sequence have same scale
        for time_iter in xrange(len(path2sample)):
            data_list[time_iter] = cv2.resize(data_list[time_iter], None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            label_list[time_iter] = np.expand_dims(cv2.resize(label_list[time_iter], None, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), axis=2)
        
    


    # -------------------------------
    # random crop and padding
    # -------------------------------

    input_data_list = []

    data_crop, label_crop = randomCropAndPadding(data_list, label_list, config_Dataset.INPUT_SIZE[0], config_Dataset.INPUT_SIZE[1])
    for time_iter in xrange(len(path2sample)):
        input_data = {}
        input_data.update({'data': data_crop[time_iter]})
        input_data.update({'label': label_crop[time_iter]})
        input_data_list.append(input_data)

    return input_data_list


#############################################################
###                                                       ###
### function: randomCropAndPadding                        ###
###                                                       ###
#############################################################


def randomCropAndPadding(img_list, label_list, crop_height, crop_width):
        assert len(img_list) == len(label_list)
        assert img_list[0].shape[0] == label_list[0].shape[0]
        assert img_list[0].shape[1] == label_list[0].shape[1]

        img_crop_list = []
        label_crop_list = []


        # random crop
        if img_list[0].shape[0] >= crop_height and img_list[0].shape[1] >= crop_width:
            x = random.randint(0, img_list[0].shape[0] - crop_height)
            y = random.randint(0, img_list[0].shape[1] - crop_width)
            # iterate here, such that all data of one time-sequence have same padding/cropping 
            for time_iter in xrange(len(img_list)):
                img_crop_list.append(img_list[time_iter][x:x+crop_height, y:y+crop_width])
                label_crop_list.append(label_list[time_iter][x:x+crop_height, y:y+crop_width])
            #print("cropping")

        # padding
        elif img_list[0].shape[0] <= crop_height and img_list[0].shape[1] <= crop_width:
            # iterate here, such that all data of one time-sequence have same padding/cropping
            for time_iter in xrange(len(img_list)):
                img_crop = np.zeros((crop_height, crop_width, img_list[time_iter].shape[2]), dtype=np.float32)
                label_crop = np.zeros((crop_height, crop_width, label_list[0].shape[2]), dtype=np.float32)
                img_crop[0:img_list[time_iter].shape[0],0:img_list[time_iter].shape[1],:] = img_list[time_iter]
                label_crop[0:label_list[time_iter].shape[0], 0:label_list[time_iter].shape[1], :] = label_list[time_iter]
                img_crop_list.append(img_crop)
                label_crop_list.append(label_crop)
            #print("padding")

        elif img_list[0].shape[0] < crop_height and img_list[0].shape[1] > crop_width:
            y = random.randint(0, img_list[0].shape[1] - crop_width)
            # iterate here, such that all data of one time-sequence have same padding/cropping
            for time_iter in xrange(len(img_list)):
                img_crop = np.zeros((crop_height, crop_width, img_list[time_iter].shape[2]), dtype=np.float32)
                label_crop = np.zeros((crop_height, crop_width, label_list[time_iter].shape[2]), dtype=np.float32)
                img_crop[0:img_list[time_iter].shape[0],:,:] = img_list[time_iter][:, y:y+crop_width]
                label_crop[0:label_list[time_iter].shape[0], :, :] = label_list[time_iter][:, y:y+crop_width]
                img_crop_list.append(img_crop)
                label_crop_list.append(label_crop)
            
        elif img_list[0].shape[0] > crop_height and img_list[0].shape[1] < crop_width:
            x = random.randint(0, img_list[0].shape[0] - crop_height)
            # iterate here, such that all data of one time-sequence have same padding/cropping
            for time_iter in xrange(len(img_list)):
                img_crop = np.zeros((crop_height, crop_width, img_list[time_iter].shape[2]), dtype=np.float32)
                label_crop = np.zeros((crop_height, crop_width, label_list[time_iter].shape[2]), dtype=np.float32)
                img_crop = img_list[time_iter][:, y:y+crop_width] = img_list[time_iter][x:x+crop_height, :]
                label_crop = label_list[time_iter][:, y:y+crop_width] = label_list[time_iter][x:x+crop_height, :]
                img_crop_list.append(img_crop)
                label_crop_list.append(label_crop)
   
        else:
            print ("[randomCropAndPadding] Error: unsupported size")  

        return img_crop_list, label_crop_list


#############################################################
###                                                       ###
### function: substract_imageMean                         ###
###                                                       ###
#############################################################


def substract_imageMean(data, img_mean):
    """ substract image-Mean of the images"""

    return data - img_mean

#############################################################
###                                                       ###
### function: flip_data                                   ###
###                                                       ###
#############################################################


def flip_data(data_list, label_list, config):
    """ flip data and corresponding labels"""

    assert len(data_list) == len(label_list)

    for iter in xrange(len(data_list)):

            # flip data

            data_list[iter] = cv2.flip(data_list[iter],1)

            # flip groundtruth

            label_list[iter] = cv2.flip(label_list[iter],1)


