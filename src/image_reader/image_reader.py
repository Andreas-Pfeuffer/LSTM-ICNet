
import os
import sys
import time
import Queue
import multiprocessing as mp

import numpy as np
import cv2
import random
from random import randint


from preprocess import preprocess

use_multiprocessing = False
timeout_multiprocessing = 60 # timeout in seconds

class ImageReader():

    def __init__(self, config_ImageReader, config_Dataset, batchsize, time_sequence):
        """ init dataset """

        # --------------------------------
        # parameter initialization
        # --------------------------------

        self._cur = 0
        self._imageFlipping = config_ImageReader.FLIPPING
        self._imageScaling = config_ImageReader.SCALING

        self._config = config_ImageReader

        # --------------------------------
        # create dataset
        # --------------------------------

        self._configDataset = config_Dataset

        self.batchsize = batchsize
        self.time_sequence = time_sequence

        # create list with pathes to data

        self._dataset = self.read_labeled_image_list('', self._configDataset.PATH2DATALIST)
        self._dataset_amount = len(self._dataset)
        self._dataset_indices = np.arange(self._dataset_amount, dtype=np.uint32)

        # shuffel dataset
        
        if self._config.SHUFFLE:
            self.suffleDataset_indices()

        # -------------------------------
        # init multiprocessing
        # -------------------------------

        if use_multiprocessing:

            self.input_queue = mp.Queue() # Queue, that stores the images indices
            self.output_queue = mp.Queue() # Queue, that stores the preprocessed batches

            # start multiprocessing:
                 
            self.multiprocessing = mp.Process(target=self.multiprocessing_processElement, args=((self.input_queue),(self.output_queue),))
            self.multiprocessing.daemon = True
            self.multiprocessing.start()        # Launch reader_proc() as a separate python process

            # load first batch in Queue

            minibatch_ind = self.getNextMinibatch_indices()

            for iter in np.arange(self.batchsize):
                self.multiprocessing_set2InputQueue(map(self._dataset.__getitem__, minibatch_ind[iter]))
            


    def __del__(self):
        
        # stop multiprocessing
        if use_multiprocessing:
            self.stop_multiprocessing()


    def multiprocessing_set2InputQueue(self, path2img):
        ## Write data pathes to queue, which should be processed 
        self.input_queue.put(path2img) 


    def stop_multiprocessing(self):
        self.multiprocessing_set2InputQueue(None)            


    def multiprocessing_processElement(self, input_queue, output_queue):
        ## Read from the queue; this will be spawned as a separate Process
        while True:
            path2img = input_queue.get()         # Read from the queue and do nothing
            if (path2img == None):
                print ('Close Muliprocessing of ImageReader')
                break
            else:
                output_queue.put(preprocess(path2img, self._config, self._configDataset))

    def multiprocessing_getElement(self):
        return self.output_queue.get(timeout = timeout_multiprocessing)



    def read_labeled_image_list(self, data_dir, data_list):
        """ read in image list and create a list containing the file names"""

        f = open(data_list, 'r')
        data = []
        removed_data = 0

        for line in f:
            try:
                line_buff = line[:-1].split(' ')
            except ValueError: # Adhoc for test.
                image = label = line.strip("\n")

            image = os.path.join(data_dir, line_buff[0])
            if len(line_buff) == 3:
                depth = os.path.join(data_dir, line_buff[1])
            label = os.path.join(data_dir, line_buff[-1])
            label = label.strip()
            if not os.path.isfile(image):
                raise ValueError('Failed to find file: ' + image)

            if not os.path.isfile(label):
                raise ValueError('Failed to find file: ' + label)

            if (len(line_buff) == 3) and (not os.path.isfile(depth)):
                raise ValueError('Failed to find file: ' + depth)       

            if len(line_buff) == 3:
                data.append([image, depth, label])
            else:
                data.append([image, label])

        #print ('-----------------------------------------------------------------')
        print ('removed data: {}'.format(removed_data))

        return data

    
    def suffleDataset_indices(self):
        """ shuffle the list of indices randomly"""

        print ("------------------shuffle---------------------------")

        # shuffle batch_wise 
        if self._configDataset.USAGE_TIMESEQUENCES and (self._configDataset.TIMESEQUENCES_SLIDINGWINDOW == False):
            shuffle_indices = np.random.permutation(np.arange(self._dataset_amount/self._configDataset.TIMESEQUENCE_LENGTH))
            buff1 = np.kron(shuffle_indices, self._configDataset.TIMESEQUENCE_LENGTH * np.ones(self._configDataset.TIMESEQUENCE_LENGTH, dtype=int))
            buff2 = np.kron(np.ones(self._dataset_amount/self._configDataset.TIMESEQUENCE_LENGTH, dtype=int), xrange(self._configDataset.TIMESEQUENCE_LENGTH))
            self._dataset_indices = buff1 + buff2
        # shuffling for tiemsequences with sliding window
        elif self._configDataset.USAGE_TIMESEQUENCES and self._configDataset.TIMESEQUENCES_SLIDINGWINDOW:
            print ('[suffleDataset_indices] ERROR: time sequences shuffling is not implemented - the dataset is not shuffled!!!')
        # shuffle every element
        else:
            self._dataset_indices = np.random.permutation(np.arange(self._dataset_amount))

        self._cur = 0


    def getNextMinibatch_indices(self):
        """return the indices for the next minibatch"""

        if self._cur + self.batchsize * self.time_sequence > len(self._dataset_indices):
            if self._config.SHUFFLE:
                self.suffleDataset_indices()
            else:
                self._cur = 0

        indices = []
        for iter in xrange(self.batchsize):
            indices.append(self._dataset_indices[(self._cur + self.time_sequence*iter): (self._cur + self.time_sequence*(iter+1))])
        self._cur =  self._cur + self.batchsize * self.time_sequence

        return indices


     

    def getNextMinibatch(self):
        """ return the next minibatch"""

        # ---------------------------------------
        # determine indices of next minibatch
        # ---------------------------------------

        minibatch_ind = self.getNextMinibatch_indices()
        #print (minibatch_ind)

        # ---------------------------------------
        # preprocessing of each data sample
        # ---------------------------------------


        training_batch = self.getInputBlob(minibatch_ind)
            

        return training_batch


    def getInputBlob(self, minibatch_ind):
        """ creates the next minibatch for training/testing/... """
        
        data_array = []
        data_scale_array = []
        label_array = []
        input_array = []

        # ---------------------------------------
        # preprocessing of each data sample
        # ---------------------------------------

        for iter in np.arange(self.batchsize):

            # fetch next sample

            if use_multiprocessing: 
                # fetch sample from queue
                try:
                    input_data = self.multiprocessing_getElement()
                except Queue.Empty:
                    print ('[WARNING] Batch-Queue is empty! Use classical input-Reader instead!')
                    input_data = preprocess(map(self._dataset.__getitem__, minibatch_ind[iter]), self._config, self._configDataset)
                    
                # set new sample for preprocessing
                self.multiprocessing_set2InputQueue(map(self._dataset.__getitem__, minibatch_ind[iter]))
            else:
                input_data = preprocess(map(self._dataset.__getitem__, minibatch_ind[iter]), self._config, self._configDataset)

            for time_iter in xrange(len(input_data)):
                input_array.append(input_data[time_iter])


        # ---------------------------------------
        # put data into minibatch (data)
        # ---------------------------------------

        training_batch = {}
        imInfo_exist = False

        for key in input_data[0].iterkeys():
            if (key == 'imInfo'):
                imInfo_exist = True
            elif (key == 'gt_object'):
                # Note: only batchsize of 1 is supported until now!
                data = input_array[iter][key]
                blob_label = np.zeros((len(input_array), data.shape[0], 5), dtype=np.float32)  
                blob_label[0, :, :] = data
                training_batch.update({'blob_label': blob_label})
                
            else:
                data_array = []
                for iter in np.arange(len(input_array)):
                    data_array.append(input_array[iter][key]) 

                max_size = np.array([data.shape for data in data_array]).max(axis=0)       
                blob_data = np.zeros((len(input_array), max_size[0], max_size[1], data.shape[-1]), dtype=np.float32)
                for iter in np.arange(len(input_array)):
                    blob_data[iter, 0:data_array[iter].shape[0], 0:data_array[iter].shape[1], :] = data_array[iter]

                training_batch.update({'blob_'+key: blob_data})

        if imInfo_exist:
            blob_imInfo = np.zeros((len(input_array), 3), dtype=np.float32)  
            for iter in np.arange(len(input_array)):
                if input_array[iter]['imInfo'][0] == input_array[iter]['imInfo'][1]:
                    blob_imInfo[iter, :] = np.array([training_batch['blob_data'].shape[1], training_batch['blob_data'].shape[2], input_array[iter]['imInfo'][0]], dtype=np.float32)
                else:
                    raise NotImplementedError
            training_batch.update({'blob_imInfo': blob_imInfo})


        #print (training_batch['blob_data'].shape)

        return training_batch








