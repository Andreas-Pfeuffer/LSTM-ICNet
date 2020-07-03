##############################################################################################
####                                                                                      ####
#### data augmentation method "Robust Learning Method", for Video Segmentations           ####
#### use brightness modifications                                                         ####
####                                                                                      ####
##############################################################################################


import cv2
import numpy as np
import random


def dataAugmentation_RLMvideoBrightnessModification(imageSequence):

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

        # choose alpha and beta

        buff = np.random.randint(899, size=2) + 1
        buff2 = np.random.randint(2, size=2)

        if buff2[0] == 0:
            alpha = 0.1 + buff[0]/1000.0
        else:
            alpha = 1.0 + buff[0]/100

        beta = 0

        # Brightness and contrast adjustments 

        imageSequence[disturb_image] = cv2.convertScaleAbs(imageSequence[disturb_image], alpha=alpha, beta=beta) 

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

        # choose alpha and beta

        buff = np.random.randint(899, size=2) + 1
        buff2 = np.random.randint(2, size=2)

        if buff2[0] == 0:
            alpha = 0.1 + buff[0]/1000.0
        else:
            alpha = 1.0 + buff[0]/100

        beta = 0

        # Brightness and contrast adjustments 

        for iter in disturb_image:
            imageSequence[iter] = cv2.convertScaleAbs(imageSequence[iter], alpha=alpha, beta=beta) 


    # ------------------------------------------------------------------
    # case 3: all frames of the sequence is disturbed
    # ------------------------------------------------------------------

    elif (disturbance_type == 3):

        disturb_image = np.arange(sequence_length)

        # choose alpha and beta

        buff = np.random.randint(899, size=2) + 1
        buff2 = np.random.randint(2, size=2)

        if buff2[0] == 0:
            alpha = 0.1 + buff[0]/1000.0
        else:
            alpha = 1.0 + buff[0]/100

        beta = 0

        # Brightness and contrast adjustments 

        for iter in disturb_image:
            imageSequence[iter] = cv2.convertScaleAbs(imageSequence[iter], alpha=alpha, beta=beta) 


    return imageSequence

