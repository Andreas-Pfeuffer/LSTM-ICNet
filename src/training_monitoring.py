from __future__ import print_function

import sys
import time
import argparse
import os
import shutil
import math
import matplotlib.pyplot as plt
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


def init_trainingMonitoring(training_monitoring, config):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(training_monitoring[:,0], training_monitoring[:,1], 'r', label='Training - mIoU')
    plt.plot(training_monitoring[:,0], training_monitoring[:,2], 'b', label='Validation - mIoU')
    plt.plot(training_monitoring[:,0], training_monitoring[:,4], 'm', label='Training - accuracy')
    plt.plot(training_monitoring[:,0], training_monitoring[:,5], 'c', label='Validation - accuracy')
    plt.axis([0,  config.NUM_STEPS + 10, 0, 1])
    plt.ylabel('mIoU/acc')
    plt.xlabel('iterations')
    plt.title('Training/Validation Accurarcy')
    plt.legend(loc='upper left', shadow=False)

    # Set Grid Properties
    if config.NUM_STEPS > 20000:
        major_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 10000)
        minor_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 2500)
    elif config.NUM_STEPS > 2000:
        major_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 1000)
        minor_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 250)
    else:
        major_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 100)
        minor_ticks_x = np.arange(0,  config.NUM_STEPS + 10, 25)
    major_ticks_y = np.arange(0, 1, 0.10)
    minor_ticks_y = np.arange(0, 1, 0.05)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.plot(training_monitoring[0,0], training_monitoring[0,1], '-ro', label='Training - mIoU')
    plt.plot(training_monitoring[0,0], training_monitoring[0,2], '-bo', label='Validation - mIoU')
    plt.plot(training_monitoring[0,0], training_monitoring[0,4], '-mo', label='Training - accuracy')
    plt.plot(training_monitoring[0,0], training_monitoring[0,5], '-co', label='Validation - accuracy')

    plt.draw() 
    plt.pause(0.000001)


def update_TrainingMonitoring(training_monitoring, iter, best_epoch=-99):

    plt.plot(training_monitoring[0:iter+1,0], training_monitoring[0:iter+1,1], '-ro', label='Training - mIoU')
    plt.plot(training_monitoring[0:iter+1,0], training_monitoring[0:iter+1,2], '-bo', label='Validation - mIoU')
    plt.plot(training_monitoring[0:iter+1,0], training_monitoring[0:iter+1,4], '-mo', label='Training - accuracy')
    plt.plot(training_monitoring[0:iter+1,0], training_monitoring[0:iter+1,5], '-co', label='Validation - accuracy')
    if best_epoch > 0:
        plt.plot(training_monitoring[best_epoch,0], training_monitoring[best_epoch,2], '-go', label='best Validation')
    plt.draw() 
    plt.pause(0.000001)



def hold_TrainingMonitoring():
    
    plt.show()
