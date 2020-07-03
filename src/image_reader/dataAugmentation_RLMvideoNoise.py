##############################################################################################
####                                                                                      ####
#### data augmentation method "Robust Learning Method", for Video Segmentations           ####
#### use random selcted gausian and salt and pepper noise                                 ####
####                                                                                      ####
##############################################################################################


import cv2
import numpy as np
import random


def dataAugmentation_RLMvideoNoise(imageSequence):

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

        # choose noise type:

        noise_type = np.random.randint(2, size=1)

        # gaussian noise

        if (noise_type == 0):
            std = np.random.uniform(0.0, 0.25, 1) * 255.0
            noise = np.random.normal(0.0,std, imageSequence[disturb_image].shape[0:2])

            imageSequence[disturb_image][:,:,0] = np.clip(imageSequence[disturb_image][:,:,0] + noise, 0, 255)
            imageSequence[disturb_image][:,:,1] = np.clip(imageSequence[disturb_image][:,:,1] + noise, 0, 255)
            imageSequence[disturb_image][:,:,2] = np.clip(imageSequence[disturb_image][:,:,2] + noise, 0, 255)

        # salt & pepper

        elif (noise_type == 1):

            # determine random drop probability
            prob = np.random.uniform(0.0, 0.33, 1)

            # salt & pepper

            imageSequence[disturb_image] = salt_n_pepper(imageSequence[disturb_image], prob=prob)

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

        # choose noise type:

        noise_type = np.random.randint(2, size=1)

        # gaussian noise

        if (noise_type == 0):
            std = np.random.uniform(0.0, 0.25, 1) * 255.0
            noise = np.random.normal(0.0,std, imageSequence[0].shape[0:2])

            for iter in disturb_image:
                imageSequence[iter][:,:,0] = np.clip(imageSequence[iter][:,:,0] + noise, 0, 255)
                imageSequence[iter][:,:,1] = np.clip(imageSequence[iter][:,:,1] + noise, 0, 255)
                imageSequence[iter][:,:,2] = np.clip(imageSequence[iter][:,:,2] + noise, 0, 255)

        # salt & pepper

        elif (noise_type == 1):

            # determine random drop probability
            prob = np.random.uniform(0.0, 0.33, 1)

            # salt & pepper

            for iter in disturb_image:
                imageSequence[iter] = salt_n_pepper(imageSequence[iter], prob=prob)

    # ------------------------------------------------------------------
    # case 3: all frames of the sequence is disturbed
    # ------------------------------------------------------------------

    elif (disturbance_type == 3):

	disturb_image = np.arange(sequence_length)

        # choose noise type:

        noise_type = np.random.randint(2, size=1)

        # gaussian noise

        if (noise_type == 0):
            std = np.random.uniform(0.0, 0.25, 1) * 255.0
            noise = np.random.normal(0.0,std, imageSequence[0].shape[0:2])

            for iter in disturb_image:
                imageSequence[iter][:,:,0] = np.clip(imageSequence[iter][:,:,0] + noise, 0, 255)
                imageSequence[iter][:,:,1] = np.clip(imageSequence[iter][:,:,1] + noise, 0, 255)
                imageSequence[iter][:,:,2] = np.clip(imageSequence[iter][:,:,2] + noise, 0, 255)

        # salt & pepper

        elif (noise_type == 1):

            # determine random drop probability
            prob = np.random.uniform(0.0, 0.33, 1)

            # salt & pepper

            for iter in disturb_image:
                imageSequence[iter] = salt_n_pepper(imageSequence[iter], prob=prob)


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

def to_std_uint8(img):
    # Properly handles the conversion to uint8
    img = cv2.convertScaleAbs(img, alpha = (255.0/1.0))
    return img

def to_std_float(img):
    #Converts img to 0 to 1 float to avoid wrapping that occurs with uint8
    img.astype(np.float16, copy = False)
    img = np.multiply(img, (1.0/255.0)) 
    return img    

def salt_n_pepper(img, prob = 0.001):
    # Convert img1 to 0 to 1 float to avoid wrapping that occurs with uint8
    img = to_std_float(img)
     
    # Generate noise to be added to the image. We are interested in occurrences of high
    # and low bounds of pad. Increased pad size lowers occurence of high and low bounds.
    # These high and low bounds are converted to salt and pepper noise later in the
    # function. randint is inclusive of low bound and exclusive of high bound.
    noise = np.random.randint(10000, size = (img.shape[0], img.shape[1], 1))

    threth_down = prob * 10000.0
    threth_up = (1.0 - prob) * 10000.0
     
    # Convert high and low bounds of pad in noise to salt and pepper noise then add it to
    # our image. 1 is subtracted from pad to match bounds behaviour of np.random.randint.
    img = np.where(noise < threth_down, 0, img)
    img = np.where(noise > threth_up, 1, img)
     
    # Properly handles the conversion from float16 back to uint8
    img = to_std_uint8(img)

 
    return img
