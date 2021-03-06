from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np
import cv2
from tqdm import trange
import argparse
import shutil

from tensorflow.python import pywrap_tensorflow 
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import itertools
import matplotlib.pyplot as plt

import _init_paths
from config import init_config
from image_reader import ImageReader
from model import get_model
from tools import decode_labels

# Use an end-to-end tensorflow model.
# Note: models in E2E tensorflow mode have only been tested in feed-forward mode,
#       but these models are exportable to other tensorflow instances as GraphDef files.
USE_E2E_TF = True


def get_arguments():
    parser = argparse.ArgumentParser(description="SemanticLabelingToolbox - for training and validation")

    parser.add_argument("--model", type=str, default='',
                        help="Network architecture.")
    parser.add_argument("--model-variant", type=str, default='',
                        help="Variant of network architecture (for models that support this)")
    parser.add_argument("--pretrained-model", type=str, default='',
                        help="pretrained model")
    parser.add_argument("--dataset", type=str, default='',
                        help="dataset to train") 
    parser.add_argument("--plot_confusionMatrix", type=bool, default=True,
                        help="plot confusion Matrix (only supported for semantic segmentation") 
    parser.add_argument("--architecture", type=str, default='semantic_segmentation',
                        help="choose architecture ('semantic_segmentation' or'object_detection' )")
    parser.add_argument("--evaluation_set", type=str, default='val',
                        help="evaluation set: whether to evaluate on validation (val) or training (train) set")
    parser.add_argument("--weather", type=str, default='all_train',
                        help="weather_condition: select between 'all', 'sunny', 'night', 'foggy', 'snowy', 'rainy'")

    return parser.parse_args()


# code based on https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure()
    if normalize:
        confusion_matrix = 100 * confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))




def evaluate(config, evaluation_set='val', plot_confusionMatrix=False):


    # --------------------------------------------------------------------
    # init network
    # --------------------------------------------------------------------

    tf.compat.v1.reset_default_graph()

    # define input placeholders

    input_placeholder = {}

    input_placeholder.update({'is_training': tf.compat.v1.placeholder(dtype=tf.bool, shape=())})
    
    if config.ARCHITECTURE == 'semantic_segmentation' :
        batch_size = config.BATCH_SIZE * config.TIMESEQUENCE_LENGTH
        # Search for available GPUs: the result is a list of device ids like `['/gpu:0', '/gpu:1']`
        devices = get_available_gpus()
        print ("found devices: ", devices)
        num_GPU = len(devices)
        if (num_GPU) == 0:
            num_GPU = 1 # CPU support!
        # min 1 sample should be applied on a GPU
        if (config.BATCH_SIZE < num_GPU):
            num_GPU = config.BATCH_SIZE

        image_placeholder = []
        label_placeholder = []
        for iter in range(num_GPU):
            if (iter == (num_GPU -1)):
                batch_size_local = batch_size - (num_GPU - 1) * (batch_size // num_GPU)
            else:
                batch_size_local = batch_size // num_GPU 
            print ('batch_size /gpu:{} : {}'.format(iter, num_GPU))

            image_placeholder.append(tf.compat.v1.placeholder(dtype=tf.float32,
                                                                shape=(batch_size_local,
                                                                       config.DATASET_TRAIN.INPUT_SIZE[0],
                                                                       config.DATASET_TRAIN.INPUT_SIZE[1],
                                                                       config.DATASET_TRAIN.NUM_CHANNELS)))
            label_placeholder.append(tf.compat.v1.placeholder(dtype=tf.float32,
                                                                shape=(batch_size_local,
                                                                       config.DATASET_TRAIN.INPUT_SIZE[0],
                                                                       config.DATASET_TRAIN.INPUT_SIZE[1],
                                                                       1)))

        input_placeholder.update({'image_batch': image_placeholder})
        input_placeholder.update({'label_batch': label_placeholder})
    else:
        print ('[ERROR] network architecture does not exist!!! Please check your spelling!')
        raise NotImplementedError


    # load network architecture

    if config.ARCHITECTURE == 'semantic_segmentation' :
        model = get_model(config.MODEL)
        net = model({'data': input_placeholder['image_batch'], 'is_training': input_placeholder['is_training']}, 
                    is_training=input_placeholder['is_training'],
                    evaluation=tf.logical_not(input_placeholder['is_training']), 
                    #is_inference=True,
                    num_classes=config.DATASET_TRAIN.NUM_CLASSES,
                    filter_scale=config.FILTER_SCALE,
                    timeSequence=config.TIMESEQUENCE_LENGTH,
                    variant=config.MODEL_VARIANT)
    else:
        print ('[ERROR] network architecture does not exist!!! Please check your spelling!')
        raise NotImplementedError
    
    # --------------------------------------------------------------------
    # determine evaluation metric
    # --------------------------------------------------------------------

    if config.ARCHITECTURE == 'semantic_segmentation' :

        list_raw_gt = []
        list_pred_flattern_mIoU = []

        for iter_gpu in range(len(input_placeholder['image_batch'])):
            with tf.device('/gpu:%d' % iter_gpu):
                if config.MODEL == 'SegNet_BN' or config.MODEL == 'SegNet_BN_encoder' or config.MODEL == 'SegNet_BN_decoder' or config.MODEL == 'SegNet_BN_encoderDecoder':
                    raw_output = net.layers['output'][iter_gpu]
                    
                    raw_output_up = tf.argmax(raw_output, axis=3, output_type=tf.int32)
                    raw_pred_mIoU = tf.expand_dims(raw_output_up, dim=3)
                else:  # ICNet
                    ori_shape = config.DATASET_TRAIN.INPUT_SIZE #??
                    raw_output = net.layers['output'][iter_gpu]
                
                    raw_output_up = tf.compat.v1.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
                    raw_output_up = tf.argmax(raw_output_up, axis=3, output_type=tf.int32)
                    raw_pred_mIoU = tf.expand_dims(raw_output_up, dim=3)
        
                # determine mIoU
                
                if config.USAGE_TIMESEQUENCES:  # evaluate only last image of time sequence
                    pred_of_interest = np.array(range(config.BATCH_SIZE),dtype=np.int32)*config.TIMESEQUENCE_LENGTH + config.TIMESEQUENCE_LENGTH - 1
                    pred_flatten_mIoU = tf.reshape(tf.gather(raw_pred_mIoU, pred_of_interest), [-1,])
                    raw_gt = tf.reshape(tf.gather(input_placeholder['label_batch'][iter_gpu],pred_of_interest), [-1,])
                else:                           # evaluate all images of batch size
                    pred_flatten_mIoU = tf.reshape(raw_pred_mIoU, [-1,])
                    raw_gt = tf.reshape(input_placeholder['label_batch'][iter_gpu], [-1,])

                list_raw_gt.append(raw_gt)
                list_pred_flattern_mIoU.append(pred_flatten_mIoU)

        # combine output of different GPUs
        with tf.device('/gpu:%d' % 0):
            all_raw_gt = tf.reshape(tf.concat(list_raw_gt, -1), [-1, ])
            all_pred_flatten_mIoU = tf.reshape(tf.concat(list_pred_flattern_mIoU,-1) , [-1,])
        
            indices_mIoU = tf.squeeze(tf.where(tf.less_equal(raw_gt, config.DATASET_TRAIN.NUM_CLASSES - 1)), 1)
            gt_mIoU = tf.cast(tf.gather(raw_gt, indices_mIoU), tf.int32)
            pred_mIoU = tf.gather(pred_flatten_mIoU, indices_mIoU)

        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_mIoU, gt_mIoU, num_classes=config.DATASET_VAL.NUM_CLASSES)

        # create colored image

        pred_color = decode_labels(pred_flatten_mIoU, config.DATASET_VAL.INPUT_SIZE[0:2], config.DATASET_VAL.NUM_CLASSES)         

        # deterimine confusing matrix

        if plot_confusionMatrix:
            # Create an accumulator variable to hold the counts
            confusion = tf.Variable( tf.zeros([config.DATASET_VAL.NUM_CLASSES, config.DATASET_VAL.NUM_CLASSES], dtype=tf.int64 ), name='confusion', 
                                     collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES] )
            # Compute a per-batch confusion
            batch_confusion = tf.math.confusion_matrix(tf.reshape(gt_mIoU, [-1]), tf.reshape(pred_mIoU, [-1]), num_classes=config.DATASET_VAL.NUM_CLASSES, name='batch_confusion')
            # Create the update op for doing a "+=" accumulation on the batch
            confusion_update = confusion.assign( confusion + tf.cast(batch_confusion, dtype=tf.int64) )


    # -----------------------------------------
    # init session
    # -----------------------------------------

    # Set up tf session and initialize variables.

    sessConfig = tf.compat.v1.ConfigProto()
    sessConfig.gpu_options.allow_growth = True
    # use only a fraction of gpu memory, otherwise the TensorRT-test has not enough free GPU memory for execution
    sessConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=sessConfig)
    init = tf.compat.v1.global_variables_initializer()
    local_init = tf.compat.v1.local_variables_initializer()
    
    sess.run(init)
    sess.run(local_init)

    # load checkpoint file
    
    print (config.EVALUATION.MODELPATH)
    ckpt = tf.compat.v1.train.get_checkpoint_state(config.EVALUATION.MODELPATH)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
        load(loader, sess, ckpt.model_checkpoint_path)

    else:
        print('No checkpoint file found.')

    
    # `sess.graph` provides access to the graph used in a <a href="./../api_docs/python/tf/Session"><code>tf.Session</code></a>.
    writer = tf.compat.v1.summary.FileWriter("/tmp/tensorflow_graph", tf.compat.v1.get_default_graph())



    # --------------------------------------------------------------------
    # Evaluate - Iterate over training steps.
    # --------------------------------------------------------------------


    # evaluate training or validation set

    if evaluation_set == "val":
        imagereader_val = ImageReader(config.IMAGEREADER.VAL, config.DATASET_VAL, config.BATCH_SIZE, config.TIMESEQUENCE_LENGTH)
    elif evaluation_set == "train":
        imagereader_val = ImageReader(config.IMAGEREADER.VAL, config.DATASET_TRAIN, config.BATCH_SIZE, config.TIMESEQUENCE_LENGTH)
    elif evaluation_set == "test":
        imagereader_val = ImageReader(config.IMAGEREADER.VAL, config.DATASET_TEST, config.BATCH_SIZE, config.TIMESEQUENCE_LENGTH)
    elif evaluation_set == "all":
        imagereader_val = ImageReader(config.IMAGEREADER.VAL, config.DATASET_ALL, config.BATCH_SIZE, config.TIMESEQUENCE_LENGTH)
    else:
        print ("Dataset {} does not exist!".format(evaluation_set))



    filename_memory = ""
    filename_count = 0
    average_inference_time = 0

    # --------------------------------------
    # perform evaluation - semantic segmentation
    # --------------------------------------

    if config.ARCHITECTURE == 'semantic_segmentation':
        if config.TIMESEQUENCES_SLIDINGWINDOW:  # use time sequences
            for step in trange(int(imagereader_val._dataset_amount - config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH + 1), desc='inference', leave=True):
                #start_time = time.time()

                training_batch = imagereader_val.getNextMinibatch()

                feed_dict = {input_placeholder['is_training']: False}

                for iter_GPU in range(len(input_placeholder['image_batch'])):        
                    num_GPU = len(input_placeholder['image_batch'])
                    batch_size = training_batch['blob_data'].shape[0]
                    batch_size_local = batch_size // num_GPU
                    if (iter_GPU == (num_GPU -1)):
                        batch_size_act = batch_size - (num_GPU - 1) * (batch_size // num_GPU)
                    else:
                        batch_size_act = batch_size // num_GPU

                    feed_dict.update({input_placeholder['image_batch'][iter_GPU]: training_batch['blob_data'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:],
                                      input_placeholder['label_batch'][iter_GPU]: training_batch['blob_label'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:]})

                start_time = time.time()
                
                if plot_confusionMatrix:
                    sess.run([update_op, confusion_update], feed_dict=feed_dict)
                else:
                    sess.run([update_op], feed_dict=feed_dict)

                duration = time.time() - start_time
                average_inference_time += duration

                # save image
                prediction = sess.run([pred_color], feed_dict=feed_dict)  
                predi = np.array(prediction[0][0,:,:,:]).astype(dtype=np.uint8)

                img = cv2.imread(imagereader_val._dataset[step][0])
                if imagereader_val._configDataset.USE_IMAGE_ROI:
                    img = img[imagereader_val._configDataset.IMAGE_ROI_MIN_X:imagereader_val._configDataset.IMAGE_ROI_MAX_X, 
                              imagereader_val._configDataset.IMAGE_ROI_MIN_Y:imagereader_val._configDataset.IMAGE_ROI_MAX_Y,:]
                cv2.addWeighted(predi, config.INFERENCE.OVERLAPPING_IMAGE, img, 1 - config.INFERENCE.OVERLAPPING_IMAGE, 0, img) 

                buff = imagereader_val._dataset[step][-1]
                buff = buff.split('/')
                filename = buff[-1].split('.')[0]
                if filename_memory == buff[-1].split('.')[0]:
                    filename_count += 1
                    filename = buff[-1].split('.')[0] + "_" + str(filename_count) + ".png"
                else:
                    filename_memory = buff[-1].split('.')[0]
                    filename = buff[-1].split('.')[0] + "_0.png"
                    filename_count = 0
                    
                cv2.imwrite(config.INFERENCE.SAVEDIR_IMAGES+filename, predi[:,:,(2,1,0)])
                cv2.imwrite(config.INFERENCE.SAVEDIR_IMAGES+filename, img[:,:,(2,1,0)])
            # determine average time
            average_inference_time = average_inference_time / int(imagereader_val._dataset_amount - config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH + 1)
        else:        # do not use time sequences (normal evaluation)
            # TODO parameters for flickering evaluation

            flickering_sum = 0
            flickering_img_size = 0
            flickering_sum1 = 0
            flickering_img_size1 = 0
    
            for step in trange(int(imagereader_val._dataset_amount/(config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH)), desc='inference', leave=True):

                training_batch = imagereader_val.getNextMinibatch()

                feed_dict = {input_placeholder['is_training']: False}

                for iter_GPU in range(len(input_placeholder['image_batch'])):        
                    num_GPU = len(input_placeholder['image_batch'])
                    batch_size = training_batch['blob_data'].shape[0]
                    batch_size_local = batch_size // num_GPU
                    if (iter_GPU == (num_GPU -1)):
                        batch_size_act = batch_size - (num_GPU - 1) * (batch_size // num_GPU)
                    else:
                        batch_size_act = batch_size // num_GPU

                    # for depth images, if result should be shown in camera image

                    if True:
                        feed_dict.update({input_placeholder['image_batch'][iter_GPU]: training_batch['blob_data'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:],
                                          input_placeholder['label_batch'][iter_GPU]: training_batch['blob_label'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:]})
                    else:
                        training_batch['blob_data'][:,:,:,1] = training_batch['blob_data'][:,:,:,-1]
                        training_batch['blob_data'][:,:,:,2] = training_batch['blob_data'][:,:,:,-1]

                        feed_dict.update({input_placeholder['image_batch'][iter_GPU]: training_batch['blob_data'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,1:4],
                                          input_placeholder['label_batch'][iter_GPU]: training_batch['blob_label'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:]})


                start_time = time.time()
                
                if plot_confusionMatrix:
                    sess.run([update_op, confusion_update], feed_dict=feed_dict)
                else:
                    sess.run([update_op], feed_dict=feed_dict)

                duration = time.time() - start_time

                # skip the first 50 images
                if step >= 50:
                        average_inference_time += duration

                # --------------------------
                # save image
                # --------------------------

                prediction = sess.run([pred_color, raw_output_up], feed_dict=feed_dict) 
                predi = np.array(prediction[0][0,:,:,:]).astype(dtype=np.uint8)
                predi_id = np.array(prediction[1][-1,...]).astype(dtype=np.uint8)

                data_index = step*config.TIMESEQUENCE_LENGTH + config.TIMESEQUENCE_LENGTH - 1

                

                buff = imagereader_val._dataset[data_index][-1]
                buff = buff.split('/')
                filename = buff[-1].split('.')[0]
                if filename_memory == buff[-1].split('.')[0]:
                    filename_count += 1
                    filename = buff[-1].split('.')[0] + "_" + str(filename_count).zfill(6) + ".png"
                else:
                    filename_memory = buff[-1].split('.')[0]
                    filename = buff[-1].split('.')[0] + "_000000.png"
                    filename_count = 0

                img = cv2.imread(imagereader_val._dataset[data_index][0])
                if imagereader_val._configDataset.USE_IMAGE_ROI:
                    img = img[imagereader_val._configDataset.IMAGE_ROI_MIN_X:imagereader_val._configDataset.IMAGE_ROI_MAX_X, 
                              imagereader_val._configDataset.IMAGE_ROI_MIN_Y:imagereader_val._configDataset.IMAGE_ROI_MAX_Y,:]

                # label images for video                

                cv2.addWeighted(predi, config.INFERENCE.OVERLAPPING_IMAGE, img, 1 - config.INFERENCE.OVERLAPPING_IMAGE, 0, img) 

                    
                cv2.imwrite(config.INFERENCE.SAVEDIR_IMAGES+ 'pred_' + filename, predi[:,:,(2,1,0)])
                cv2.imwrite(config.INFERENCE.SAVEDIR_IMAGES+ 'overlay_' + filename, img[:,:,(2,1,0)])



                # --------------------------
                # flickering evaluation
                # --------------------------

                # to measure flickering between non-ground-truth classes, multiply it with the predicted result (add 1, since True * class 0 = 0!)
                diff_img = np.array((predi_id != training_batch['blob_label'][-1,:,:,0]),np.float32) * np.array(training_batch['blob_label'][-1,:,:,0]+1, np.float32)
                diff_img1 = np.array(predi_id, np.float32)

                if step > 0:   # skip step = 0, since there is no reference image
                    flickering = np.sum((diff_img != prediction_old))
                    flickering_sum += flickering
                    flickering_img_size += predi_id.shape[0] * predi_id.shape[1] 

                    flickering1 = np.sum((diff_img1 != prediction_old1))
                    flickering_sum1 += flickering1
                    flickering_img_size1 += predi_id.shape[0] * predi_id.shape[1] 

                prediction_old = diff_img
                prediction_old1 = diff_img1
                    

            # determine average time
            average_inference_time = average_inference_time / int(imagereader_val._dataset_amount/(config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH)-50)

        mIoU_value = sess.run(mIoU)

        print ('flickering_sum: ', flickering_sum)
        print ('FP: ', float(flickering_sum)/float(flickering_img_size))
        print ('--------------')
        print ('flickering_sum1: ', flickering_sum1)
        print ('FIP: ', float(flickering_sum1)/float(flickering_img_size1))
        print ('--------------')        
        print (float(flickering_sum)/float(flickering_img_size), float(flickering_sum1)/float(flickering_img_size1))
        print ('--------------')

        if plot_confusionMatrix:
            confusion_matrix = sess.run(confusion)

            # print Accuracy:
            np.set_printoptions(linewidth=np.inf) #150)
            acc_value = float(np.sum(np.diag(confusion_matrix)))/ float(np.sum(confusion_matrix))
    else:
        print ('[ERROR] network architecture does not exist!!! Please check your spelling!')
        raise NotImplementedError

    # --------------------------------------------
    # create optimized pb-model
    # --------------------------------------------


    if config.FREEZEINFERENCEGRAPH.MODE:

        output_nodes = config.FREEZEINFERENCEGRAPH.OUTPUT_NODE_NAMES.split(',')
        input_nodes = config.FREEZEINFERENCEGRAPH.INPUT_NODE_NAMES.split(',')

        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, tf.compat.v1.get_default_graph().as_graph_def(), output_nodes)
    
        # Write tensorrt graph def to pb file for use in C++
        output_graph_path = config.EVALUATION.MODELPATH + '/model.pb'
        with tf.io.gfile.GFile(output_graph_path, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

        transforms = [
            'remove_nodes(op=Identity)',
            'merge_duplicate_nodes',
            'strip_unused_nodes',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'remove_device'
        ]

        optimized_graph_def = TransformGraph(frozen_graph_def, input_nodes, output_nodes, transforms)
        
        print ('inference model saved in ', output_graph_path)

    # ------------------------------------
    # apply TensorRT
    # ------------------------------------
      
    if config.FREEZEINFERENCEGRAPH.MODE:
        
        config_TensorRT = tf.ConfigProto()
        #config_TensorRT.gpu_options.per_process_gpu_memory_fraction = 0.5
        config_TensorRT.gpu_options.allow_growth = True

        converter = trt.TrtGraphConverter(
	        input_graph_def=optimized_graph_def, 
            session_config = config_TensorRT,
            max_workspace_size_bytes = 4000000000,
            nodes_blacklist=output_nodes, 
	        is_dynamic_op=False, precision_mode='FP16')
        converted_graph_def = converter.convert()

        # Write tensorrt graph def to pb file for use in C++
        output_graph_TensorRT_path = config.EVALUATION.MODELPATH + '/tensorrt_model.pb'
        with tf.io.gfile.GFile(output_graph_TensorRT_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())

        print ('TensorRT-model saved in ', output_graph_TensorRT_path)


    # --------------------------------------------
    # close session
    # --------------------------------------------


    writer.close()
    sess.close()
    tf.compat.v1.reset_default_graph()


    # ------------------------------------------
    # define inference-function for model.pb
    # ------------------------------------------

    def determineInferenceTime(path2frozenmodel):

        # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(path2frozenmodel, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        for node in graph_def.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignMovingAvg':
                node.op = 'MovingAvg'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="")

        # print output node names
        if config.FREEZEINFERENCEGRAPH.PRINT_OUTPUT_NODE_NAMES:
            for op in graph.get_operations():
                print (str(op.name))

        ##### init network

        if config.ARCHITECTURE == 'semantic_segmentation':
            image_tensor = graph.get_tensor_by_name('Placeholder_1:0')
            ouput_tensor = graph.get_tensor_by_name('ArgMax:0')

        # init session   # Note: we don't nee to initialize/restore anything. There is no Variables in this graph, only hardcoded constants 

        sess_inf = tf.compat.v1.Session(graph=graph)
        average_inference_time_frozenModel = 0.0

        for step in trange(int(imagereader_val._dataset_amount/(config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH)), desc='inference', leave=True):
            
            # get images from datastet

            training_batch = imagereader_val.getNextMinibatch()

            if config.ARCHITECTURE == 'semantic_segmentation':
                feed_dict = {image_tensor: training_batch['blob_data']}

            # apply inference
            
            start_time = time.time()

            if config.ARCHITECTURE == 'semantic_segmentation':
                sess_inf.run(ouput_tensor, feed_dict=feed_dict)

            duration_inf = time.time() - start_time

            # skip the first 50 images
            if step >= 50:
                average_inference_time_frozenModel += duration_inf

        sess_inf.close()
        average_inference_time_frozenModel = average_inference_time_frozenModel / int(imagereader_val._dataset_amount/(config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH)-50)
        
        return average_inference_time_frozenModel
    

    # ------------------------------------------
    # test model - optimized model
    # ------------------------------------------

    if config.FREEZEINFERENCEGRAPH.MODE:

        ### apply optimized model (model.pb)

        path2frozenmodel_opt = config.EVALUATION.MODELPATH + '/model.pb'
        average_inference_time_opt = determineInferenceTime(path2frozenmodel_opt)

        ### apply TensorRT model (tensorrt_model.pb)

        path2frozenmodel_tensorrt = config.EVALUATION.MODELPATH + '/tensorrt_model.pb'
        average_inference_time_tensorrt = determineInferenceTime(path2frozenmodel_tensorrt)

        print('average time optimized model: {:.2f} ms'.format(average_inference_time_opt*1000.0))
        print('average time TensorRT: {:.2f} ms'.format(average_inference_time_tensorrt*1000.0))

    # --------------------------------------------------------------------
    # Show results
    # --------------------------------------------------------------------

    print('average time: {:.2f} ms'.format(average_inference_time*1000.0))

    if plot_confusionMatrix and config.ARCHITECTURE == 'semantic_segmentation':
        # determine class-wise IoU
        buff = 0.0
        print('-------------------------------------------------------------')
        for iter in xrange(config.DATASET_VAL.NUM_CLASSES):
            if np.sum(confusion_matrix[iter,:]) == 0:  # avoid division by zero
		        IoU = 0.0
            else:
                IoU = 100.0 * confusion_matrix[iter,iter] / (np.sum(confusion_matrix[iter,:]) + np.sum(confusion_matrix[:,iter]) - confusion_matrix[iter,iter])
            buff = buff + IoU
            print('{}: {}'.format(config.DATASET_VAL.CLASSES[iter],IoU))
        print('-------------------------------------------------------------')
        print('dataset: {} - {}'.format(config.DATASET_NAME, config.DATASET_WEATHER))
        print('Accuracy: {}'.format(acc_value))
        print('mIoU: {}'.format(mIoU_value))
        print('average time: {:.2f} ms'.format(average_inference_time*1000.0))
        print('-------------------------------------------------------------')

    if plot_confusionMatrix and config.ARCHITECTURE == 'semantic_segmentation':
        print('-------------------------------------------------------------')
        print (confusion_matrix)
        print('-------------------------------------------------------------')
        #plot_confusion_matrix(confusion_matrix, config.DATASET_VAL.CLASSES)

     

    

    return mIoU_value

    
def main():  


    # get arguments

    args = get_arguments()
    
    # init config

    config = init_config(model=args.model, dataset=args.dataset, weather=args.weather, architecture=args.architecture)
    if (not args.pretrained_model == ''):
        config.EVALUATION.MODELPATH = args.pretrained_model
    if not args.model_variant == '':
        # Check if variant was given as integer
        try:
            config.MODEL_VARIANT = int(args.model_variant)
            print("Translated model variant to integer")
        except ValueError:
            config.MODEL_VARIANT = args.model_variant
        print('set model variant to: {}'.format(config.MODEL_VARIANT))

    # evaluate

    print('mIoU: {}'.format(evaluate(config, evaluation_set=args.evaluation_set, plot_confusionMatrix=args.plot_confusionMatrix)))




if __name__ == '__main__':
    main()














