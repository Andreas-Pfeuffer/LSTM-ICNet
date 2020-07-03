##############################################################################################
####                                                                                      ####
#### data augmentation method "Robust Learning Method", for Video Segmentations           ####
#### use diffrent image bluring methods                                                   ####
####                                                                                      ####
##############################################################################################


import cv2
import numpy as np
import random


def dataAugmentation_RLMvideoImageBluring(imageSequence):

    # ----------------------------
    # select augmentation method
    # ----------------------------

    # choose data Augmentation type. Generally, there are seven different modes:
    # case 0: good weather conditions, data are not modified
    # case 1: one frame of the sequence is disturbed 
    # case 2: several frames of the sequence are disturbed
    # case 3: all frames of the sequence are disturbed


    disturbance_type = np.random.randint(4, size=1)

    # get sequence length
    sequence_length = len(imageSequence)

    # ------------------------------------------------------------------
    # case 1: one frame of the sequence is disturbed
    # ------------------------------------------------------------------

    if disturbance_type == 1:


        # select image, which is disturbed

        disturb_image = np.random.randint(sequence_length, size=1)[0]

        # choose bluring type

        bluring_type = np.random.randint(2, size=1)

        # smoothing:

        if (bluring_type == 0):
            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            imageSequence[disturb_image] = cv2.filter2D(imageSequence[disturb_image],-1,kernel)

        # gaussian bluring

        elif (bluring_type == 1):
            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            imageSequence[disturb_image] = cv2.GaussianBlur(imageSequence[disturb_image],(kernel_size_x, kernel_size_y),0)

    # ------------------------------------------------------------------
    # case 2: several frames of the sequence is disturbed
    # ------------------------------------------------------------------

    elif (disturbance_type == 2):

        # select image, which is disturbed (more than one, but not all)

        if sequence_length > 1:
            disturb_amount = np.random.randint(sequence_length-1, size=1)[0]
            disturb_image = random.sample(range(sequence_length), disturb_amount+1)
        else:
            disturb_image = [0]

        # choose bluring type

        bluring_type = np.random.randint(2, size=1)

        # smoothing:

        if (bluring_type == 0):
            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            for iter in disturb_image:
                imageSequence[iter] = cv2.filter2D(imageSequence[iter],-1,kernel)

        # gaussian bluring

        elif (bluring_type == 1):

            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            for iter in disturb_image:
                imageSequence[iter] = cv2.GaussianBlur(imageSequence[iter],(kernel_size_x, kernel_size_y),0)

    # ------------------------------------------------------------------
    # case 3: all frames of the sequence is disturbed
    # ------------------------------------------------------------------

    elif (disturbance_type == 3):

        disturb_image = np.arange(sequence_length)

        # choose bluring type

        bluring_type = np.random.randint(2, size=1)

        # smoothing:

        if (bluring_type == 0):
            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            for iter in disturb_image:
                imageSequence[iter] = cv2.filter2D(imageSequence[iter],-1,kernel)

        # gaussian bluring

        elif (bluring_type == 1):

            # choose kernel size (1,3,5,7,...,49)
            buff = np.random.randint(24, size=2)
            kernel_size_x = 2*buff[0] + 1
            kernel_size_y = 2*buff[1] + 1
            kernel = np.ones((kernel_size_x, kernel_size_y),np.float32)/(kernel_size_x*kernel_size_y)

            for iter in disturb_image:
                imageSequence[iter] = cv2.GaussianBlur(imageSequence[iter],(kernel_size_x, kernel_size_y),0)


    return imageSequence

