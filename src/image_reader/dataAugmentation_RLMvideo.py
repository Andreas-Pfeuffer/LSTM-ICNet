##############################################################################################
####                                                                                      ####
#### data augmentation method "Robust Learning Method", for Video Segmentations           ####
####                                                                                      ####
##############################################################################################


import cv2
import numpy as np
import random


def dataAugmentation_RLMvideo(imageSequence):

    # ----------------------------
    # select augmentation method
    # ----------------------------

    # choose data Augmentation type. Generally, there are seven different modes:
    # case 0: good weather conditions, data are not modified
    # case 1: the camera fails in one frame of the video-sequence 
    # case 2: one frames of the video sequence are disturbed by white polygons (random points)
    # case 3: disturbed images in more than one frame

    disturbance_type = np.random.randint(4, size=1)

    # get sequence length
    sequence_length = len(imageSequence)

    # ------------------------------------------------------------------
    # case 1: the camera fails in one frame of the video-sequence
    # ------------------------------------------------------------------

    if disturbance_type == 1:
        disturb_image = np.random.randint(sequence_length, size=1)[0]
        imageSequence[disturb_image] = 255.0 * np.ones(imageSequence[disturb_image].shape)

    # ------------------------------------------------------------------
    # case 2: one frames of the video sequence are disturbed by white polygons (random points)
    # ------------------------------------------------------------------

    elif (disturbance_type) == 2:

        disturb_image = np.random.randint(sequence_length, size=1)[0]

        numberOfPoints =  int(np.random.randint(7, size=1) + 3)
        random_points = get_randomPoints(numberOfPoints, imageSequence[disturb_image])
        cv2.fillPoly(imageSequence[disturb_image], [random_points], (255,255,255))

    # ------------------------------------------------------------------
    # case 3: disturbed images in more than one frame by white polygons
    # ------------------------------------------------------------------

    elif (disturbance_type == 3):

         # select image, which is disturbed (more than one, less than all)

        if sequence_length > 1:
            disturb_amount = np.random.randint(sequence_length-1, size=1)[0]
            disturb_image = random.sample(range(sequence_length), disturb_amount+1)
        else:
            disturb_image = [0]

        # draw polygons to image:

        for iter in disturb_image:
            numberOfPoints =  int(np.random.randint(7, size=1) + 3)
            random_points = get_randomPoints(numberOfPoints, imageSequence[iter])
            cv2.fillPoly(imageSequence[iter], [random_points], (255,255,255))

    return imageSequence


    
def get_randomPoints(points_amount, img):
    x = np.random.randint(0, img.shape[1], size=(points_amount, 1))
    y = np.random.randint(0, img.shape[0], size=(points_amount, 1))
    # the first two points should be in the lower image half
    if points_amount > 2:
        buff = np.random.randint(int(img.shape[0]/2),img.shape[0], size=(2, 1))
        y[0:2] = buff
    pts = np.reshape([x, y], (2, -1)).transpose()
    return pts



