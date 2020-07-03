##############################################################################################
####                                                                                      ####
#### data augmentation method "Robust Learning Method2", for Video Segmentations          ####
####                                                                                      ####
##############################################################################################


import cv2
import numpy as np
import random

from dataAugmentation_RLMvideo import dataAugmentation_RLMvideo
from dataAugmentation_RLMvideoNoise import dataAugmentation_RLMvideoNoise
from dataAugmentation_RLMvideoImageBluring import dataAugmentation_RLMvideoImageBluring
from dataAugmentation_RLMvideoBrightnessModification import dataAugmentation_RLMvideoBrightnessModification


def dataAugmentation_RLMvideo3(imageSequence):

    # ----------------------------
    # select augmentation method
    # ----------------------------

    # choose data Augmentation type. Generally, there are seven different modes:
    # case 1: RLMvideo: fit white polygons to image sequence
    # case 2: RLMvideoNoise: fit random noise to image sequence
    # case 3: RLMvideoImageBluring: image bluring of the video sequence 
    # case 4: dataAugmentation_RLMvideoBrightnessModification: image brightness modifications

    disturbance_type = np.random.randint(4, size=1)

    # ------------------------------------------------------------------
    # case 1: RLMvideo: fit white polygons to image sequence
    # ------------------------------------------------------------------

    if disturbance_type == 0:
        return dataAugmentation_RLMvideo(imageSequence)

    # ------------------------------------------------------------------
    # case 2: RLMvideoNoise: fit random noise to image sequence
    # ------------------------------------------------------------------

    elif disturbance_type == 1:
        return dataAugmentation_RLMvideoNoise(imageSequence)

    # ------------------------------------------------------------------
    # case 3: RLMvideoImageBluring: fit random noise to image sequence
    # ------------------------------------------------------------------

    elif disturbance_type == 2:
        return dataAugmentation_RLMvideoImageBluring(imageSequence)

    # ------------------------------------------------------------------
    # case 3: RLMvideoImageBluring: fit random noise to image sequence
    # ------------------------------------------------------------------

    elif disturbance_type == 3:
        return dataAugmentation_RLMvideoBrightnessModification(imageSequence)

    # ------------------------------------------------------------------
    # else
    # ------------------------------------------------------------------ 

    return imageSequence


