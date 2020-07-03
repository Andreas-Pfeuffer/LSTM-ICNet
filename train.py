from __future__ import print_function

import _init_paths

import tensorflow as tf
import numpy as np
from tqdm import trange
import glob
import cv2
import sys
import time
import argparse
import os
import shutil
import math
import matplotlib.pyplot as plt
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

from training_monitoring import *
from config import init_config
from config_tools import *
from image_reader import ImageReader
from tools import decode_labels, prepare_label
from loss import *

from model import get_model

#use precision 16
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = '1'


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
    parser.add_argument("--result-dir", type=str, default='',
                        help="directory to save the results")
    parser.add_argument("--batch-size", type=int, default=-99,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=-99,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--optimizer", type=str, default='',
                        help="The type of optimizer to use ('MomentumOptimizer', 'AdamOptimizer' or "
                             "'GradientDescentOptimizer')")
    parser.add_argument("--freezeBN", type=str, default='',
                        help="freeze Batch-Normalization : True/False")
    parser.add_argument("--loss", type=str, default='',
                        help="loss type to use ('softmax_cross_entropy_ICNet', 'cross_entropy_median_frequency_balancing_ICNet', 'softmax_cross_entropy_SegNet', \
                                                'weightedLoss_SegNet', 'objectDetection_loss', 'softmax_cross_entropy_ICNet_GIF')")
    parser.add_argument("--max-iterations", type=int, default=-99,
                        help="Max iterations for training")
    parser.add_argument("--iter-size", type=int, default=-99,
                        help="Accumulate gradients over iter_size steps before updating weights")
    parser.add_argument("--no-evaluation", type=bool, default=False,
                        help="skip evaluation during training if true, else use default config")
    parser.add_argument("--architecture", type=str, default='semantic_segmentation',
                        help="choose architecture ('semantic_segmentation' or 'object_detection' )")
    parser.add_argument("--weather", type=str, default='all_train',
                        help="weather_condition: select between 'all_train', 'sunny', 'night', 'foggy', 'snowy', 'rainy'")
    parser.add_argument("--data-augmentation", type=str, default='None',
                        help="choose data augmentation method ('SLM' or'RLM' )")
    parser.add_argument("--dataset-scale", type=int, default='-99',
                        help="scale image_size of dataset (1: original size; 2: half_size; 4: quarter_size )")
    parser.add_argument("--config", type=str, default='None',
                        help="use saved config")

    return parser.parse_args()


def init_local_config(args, path2config='None'):

    # Initialize configs

    if args.config == 'None':
        if (not args.model == '') and (not args.dataset == '') and (args.dataset_scale > 0):
            config = init_config(model=args.model, dataset=args.dataset, weather=args.weather, architecture=args.architecture, dataset_scale=args.dataset_scale)
        elif (not args.model == '') and (not args.dataset == ''):
            config = init_config(model=args.model, dataset=args.dataset, weather=args.weather, architecture=args.architecture)
        elif (not args.model == ''):
            config = init_config(model=args.model, architecture=args.architecture)
        elif (not args.dataset == '') and (args.dataset_scale > 0):
            config = init_config(dataset=args.dataset, weather=args.weather, architecture=args.architecture, dataset_scale=args.dataset_scale)
        elif (not args.dataset == ''):
            config = init_config(dataset=args.dataset, weather=args.weather, architecture=args.architecture)
        else:
            config = init_config(architecture=args.architecture)

        # Write argument-parameters to config

        if not args.model_variant == '':
            # Check if variant was given as integer
            try:
                config.MODEL_VARIANT = int(args.model_variant)
                print("Translated model variant to integer")
            except ValueError:
                config.MODEL_VARIANT = args.model_variant
            print('set model variant to: {}'.format(config.MODEL_VARIANT))
        if args.batch_size > 0:
            config.BATCH_SIZE = args.batch_size
            print('set batch size to: {}'.format(config.BATCH_SIZE))
        if args.iter_size > 0:
            config.ITER_SIZE = args.iter_size
            print('set iter size to: {}'.format(config.ITER_SIZE))
        if args.max_iterations > 0:
            config.MAX_ITERATIONS = args.max_iterations
            print('set max_iterations to: {}'.format(config.MAX_ITERATIONS))
        if args.learning_rate > 0:
            config.OPTIMIZER.LEARNING_RATE = args.learning_rate
            print('set learning rate to: {}'.format(config.OPTIMIZER.LEARNING_RATE))
        if not args.optimizer == '':
            config.OPTIMIZER.TYPE = args.optimizer
            print('set optimizer to: {}'.format(config.OPTIMIZER.TYPE))
        if not args.loss == '':
            config.LOSS.TYPE = args.loss
            print('set optimizer to: {}'.format(config.LOSS.TYPE))
        if not args.freezeBN == '':
            if args.freezeBN == 'True' or args.freezeBN == 'true':
                config.UPDATE_MEAN_VAR = False
                config.TRAIN_BETA_GAMMA = False
                print('freeze Batchnormalization during training')
            else:
                config.UPDATE_MEAN_VAR = True
                config.TRAIN_BETA_GAMMA = True
                print('train Batchnormalization parameters')
        if not args.pretrained_model == '':
            config.PRETRAINED_MODEL = args.pretrained_model
            print('set pretrained model to: {}'.format(config.PRETRAINED_MODEL))
        if not args.result_dir == '':
            config.SNAPSHOT_DIR = args.result_dir
            print('set result directory to: {}'.format(config.SNAPSHOT_DIR))
        if args.no_evaluation:
            config.EVALUATION.DOEVALUATION = False
            config.EVALUATION.BEFORETRAINING = False
            print('disable evaluation during training')
        if not args.data_augmentation == 'None':
            config.IMAGEREADER.TRAIN.DATA_AUGMENTATION_METHOD = args.data_augmentation
            print ('set data augmentation method for training to {}'.format(config.IMAGEREADER.TRAIN.DATA_AUGMENTATION_METHOD))

    else:
        config = load_config(args.config)
        print ('[INFO] load config "{}"'.format(args.config))

    print("Using model: {}".format(config.MODEL))
    print("Using dataset: {}".format(config.DATASET_NAME))

    return config


def adjust_config(config):

    num_trainingImages = sum(1 for line in open(config.DATASET_TRAIN.PATH2DATALIST))

    # Determine amount of batches in one run through the dataset
    if config.TIMESEQUENCES_SLIDINGWINDOW:
        num_batches_in_epoch = int(num_trainingImages + 1 - config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH)
    else:
        num_batches_in_epoch = int(num_trainingImages/(config.BATCH_SIZE*config.TIMESEQUENCE_LENGTH))

    config.SAVE_PRED_EVERY = num_batches_in_epoch  # save prediction at end of each epoch

    # Determine maximum number of epochs allowed by MAX_ITERATIONS (rounding up)
    max_num_epochs = int(math.ceil(float(config.MAX_ITERATIONS)/float(num_batches_in_epoch)))
    print('Training for {} epochs'.format(max_num_epochs))
    config.NUM_STEPS = num_batches_in_epoch * max_num_epochs  # amount of training steps can get >= maximum amount of
    # steps given by user to make sure the last epoch is finished completely

    return config


def saveTrainingConfig(training_monitoring, best_epoch, config):

    # open file

    if config.ENABLE_TRAINING_PROTOCOL:
        file = open(config.SNAPSHOT_DIR + '/' + 'results.txt', 'w')

    # print training_montitoring to console/file

    if config.EVALUATION.DOEVALUATION:
        print("################################################")
        print("###       Training Monitoring                ###")
        print("################################################")
        headline_str = "{:3} {:14} \t{:11}  {:11}  {:17} \t{:11} {:11}".format("#", "Iteration", "mIoU_train",
                                                                               "mIoU_val", "Difference mIoUs",
                                                                               "Acc_train", "Acc_val")
        print(headline_str)

        if config.ENABLE_TRAINING_PROTOCOL:
            file.write("################################################\n")
            file.write("###       Training Monitoring                ###\n")
            file.write("################################################\n")
            file.write(headline_str + "\n")

        for iter in xrange(training_monitoring.shape[0]):
            if (iter == 0) or (training_monitoring[iter][0] > 0):

                line_str = "{:3d} - iter {:6.0f}: \t{:11.7%}, {:11.7%}, diff: {:11.7%} \t{:11.7%} {:11.7%}"\
                    .format(iter, training_monitoring[iter][0], training_monitoring[iter][1],
                            training_monitoring[iter][2], training_monitoring[iter][3],
                            training_monitoring[iter][4], training_monitoring[iter][5])
                print(line_str)
                if config.ENABLE_TRAINING_PROTOCOL:
                    file.write(line_str + "\n")

        print('best epoch: {} = iter {:.0f}'.format(best_epoch, training_monitoring[best_epoch][0]))
        print('mIoU training:       {:%}'.format(training_monitoring[best_epoch][1]))
        print('mIoU validation:     {:%}'.format(training_monitoring[best_epoch][2]))
        print('accuracy training:   {:%}'.format(training_monitoring[best_epoch][4]))
        print('accuracy validation: {:%}'.format(training_monitoring[best_epoch][5]))

        if config.ENABLE_TRAINING_PROTOCOL:
            file.write('\n')
            file.write('best epoch: {} = iter {:.0f}\n'.format(best_epoch, training_monitoring[best_epoch][0]))
            file.write('mIoU training:       {:%}\n'.format(training_monitoring[best_epoch][1]))
            file.write('mIoU validation:     {:%}\n'.format(training_monitoring[best_epoch][2]))
            file.write('accuracy training:   {:%}\n'.format(training_monitoring[best_epoch][4]))
            file.write('accuracy validation: {:%}\n'.format(training_monitoring[best_epoch][5]))
            file.write('\n')

    # save training conditions

    if config.ENABLE_TRAINING_PROTOCOL:
        file.write("################################################\n")
        file.write("###       Training Parameter                 ###\n")
        file.write("################################################\n")
        file.write('\n')
        file.write('max iterations:\t {}\n'.format(config.MAX_ITERATIONS))
        file.write('batch size:\t {}\n'.format(config.BATCH_SIZE))
        file.write('iter size:\t {}\n'.format(config.ITER_SIZE))
        file.write('gradient clipping: \t {}\n'.format(config.USE_GRADIENT_CLIPPING))
        file.write('maximal gradient norm: \t {}\n'.format(config.MAX_GRAD_NORM))
        file.write('timesequence_length:\t {}\n'.format(config.TIMESEQUENCE_LENGTH))
        file.write('optimizer:\t {}\n'.format(config.OPTIMIZER.TYPE))
        file.write('learning rate:\t {}\n'.format(config.OPTIMIZER.LEARNING_RATE))
        file.write('learning rate policy:\t {}\n'.format(config.OPTIMIZER.LEARNING_RATE_POLICY))
        file.write('momentum:\t {}\n'.format(config.OPTIMIZER.MOMENTUM))
        file.write('power:\t {}\n'.format(config.OPTIMIZER.POWER))
        file.write('step size:\t {}\n'.format(config.OPTIMIZER.STEP_SIZE))
        file.write('step decay:\t {}\n'.format(config.OPTIMIZER.STEP_DECAY))
        file.write('weight decay:\t {}\n'.format(config.OPTIMIZER.WEIGHT_DECAY))
        file.write('\n')
        file.write('model:\t {}\n'.format(config.MODEL))
        file.write('model variant:\t {}\n'.format(config.MODEL_VARIANT))
        if config.USE_PRETRAINED_MODEL:
            file.write('pretrained model:\t {}\n'.format(config.PRETRAINED_MODEL))
            file.write('\n')
        file.write('Training loss:\t {}\n'.format(config.LOSS.TYPE))
        if config.LOSS.TYPE == 'cross_entropy_median_frequency_balancing':
            file.write('Class Weight: ')
            file.write(" ".join(map(str, config.LOSS_CLASSWEIGHT)))
            file.write('\n')
        file.write('dataset:\t {}\n'.format(config.DATASET_NAME))
        file.write('\n')
        if not config.UPDATE_MEAN_VAR and not config.TRAIN_BETA_GAMMA:
            file.write('training policy for batch normalizations:\t Completely fixed to pretrained state (don\'t update'
                       'moving mean and moving variance and don\'t train beta and gamma)\n')
        elif not config.UPDATE_MEAN_VAR:
            file.write('training policy for batch normalizations:\t Don\'t update moving mean and moving variance\n')
        elif not config.TRAIN_BETA_GAMMA:
            file.write('training policy for batch normalizations:\t Don\'t train beta and gamma\n')
        if config.USE_GRADIENT_CLIPPING:
            file.write('Use Gradient Clipping. Maximum allowed 2-norm:\t {}\n'.format(config.MAX_GRAD_NORM))
        file.close()

    print('save trained parameter to {}'.format(config.SNAPSHOT_DIR))



def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


class Training:

    #####################################################################################################
    ###                                                                                               ###
    ###                                            init                                               ###
    ###                                                                                               ###
    #####################################################################################################

    def __init__(self, config):
        """
        Use this constructor to execute the complete training process with options as defined in passed config.

        This is the main way to do training, usually there is no need to call other functions in this class from
        outside!
        """

        # get config

        self.config = config

        # save config

        config_path = os.path.join(self.config.SNAPSHOT_DIR, "config.json")

        if not os.path.exists(self.config.SNAPSHOT_DIR):
            os.makedirs(self.config.SNAPSHOT_DIR)

        save_config(self.config, config_path)
        
        # start training

        tf.compat.v1.reset_default_graph()      

        self.serial_training()

    #####################################################################################################
    ###                                                                                               ###
    ###                                    support functions                                          ###
    ###                                                                                               ###
    #####################################################################################################

    def load(self, saver, sess, ckpt_path):
        """Load checkpoint from passed path"""
        saver.restore(sess, ckpt_path)
        print("Restored model parameters from {}".format(ckpt_path))

    def save(self, saver, sess, step):
        """Save checkpoint to snapshot directory"""
        checkpoint_path = os.path.join(self.config.SNAPSHOT_DIR, self.config.MODEL_NAME)

        if not os.path.exists(self.config.SNAPSHOT_DIR):
            os.makedirs(self.config.SNAPSHOT_DIR)
        saver.save(sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')

    # code based on https://github.com/tensorflow/tensorflow/issues/9517
    def assign_to_device(self, device, ps_device):
        """Returns a function to place variables on the ps_device.

        Args:
            device: Device for everything but variables
            ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

        If ps_device is not set then the variables will be placed on the default device.
        The best device for shared varibles depends on the platform as well as the
        model. Start with CPU:0 and then test GPU:0 to see if there is an
        improvement.
        """

        PS_OPS = [
        'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
        'MutableHashTableOfTensors', 'MutableDenseHashTable'
        ]

        def _assign(op):
            node_def = op if isinstance(op, tf.compat.v1.NodeDef) else op.node_def
            if node_def.op in PS_OPS:
                return ps_device
            else:
                return device
        return _assign

    # code based on https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    def get_available_gpus(self):
        """
            Returns a list of the identifiers of all visible GPUs.
        """
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    # code based on https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.

        :param tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
            over the devices. The inner list ranges over the different variables.
        :return: List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
        """
        average_grads = []

        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_and_vars]
            grad = tf.reduce_mean(grads, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    #####################################################################################################
    ###                                                                                               ###
    ###                                    network/optimization                                       ###
    ###                                                                                               ###
    #####################################################################################################

    def network(self, input_placeholder):
        """
        Set up network architecture (= graph) and define training loss. Prepare outputs for calculation of evaluation
        metrics.

        :param input_placeholder: Dictionary that includes the keys image_batch, label_batch and is_training.
                                  Only for object detection also include key image_info.
        :return: Loss tensor. Network outputs and ground-truths prepared for evaluation
        """

        # ---------------------------------------------------------------------
        # Get network architecture
        # ---------------------------------------------------------------------

        if self.config.ARCHITECTURE == 'semantic_segmentation':
            model = get_model(self.config.MODEL)
            net = model({'data': input_placeholder['image_batch'], 'is_training': input_placeholder['is_training']},
                        is_training=input_placeholder['is_training'],
                        evaluation=tf.logical_not(input_placeholder['is_training']),
                        num_classes=self.config.DATASET_TRAIN.NUM_CLASSES,
                        filter_scale=self.config.FILTER_SCALE,
                        timeSequence=self.config.TIMESEQUENCE_LENGTH,
                        variant=self.config.MODEL_VARIANT)
        else:
            print('[ERROR] network architecture does not exist!!! Please check your spelling!')
            raise NotImplementedError

        # ---------------------------------------------------------------------
        # Define training loss
        # ---------------------------------------------------------------------

        loss_array = [] 


        for iter_gpu in range(len(input_placeholder['image_batch'])):
            with tf.device('/gpu:%d' % iter_gpu):
                # Cannot use self.config.BATCH_SIZE, as that doesn't cover reduced batch size for multi-GPU processing
                # Instead, recover batch size from image_batch
                batch_size = input_placeholder['image_batch'][iter_gpu].shape.as_list()[0]
                # Even in multi-GPU processing, each GPU must process complete time sequences
                assert batch_size % self.config.TIMESEQUENCE_LENGTH == 0
                batch_size = batch_size // self.config.TIMESEQUENCE_LENGTH

                if self.config.ARCHITECTURE == 'semantic_segmentation':
                    try:
                        label_batch = input_placeholder['label_batch'][iter_gpu]
                        reduced_loss = net.get_trainingLoss(label_batch, self.config, iter_gpu)
                    except:
                        raise TypeError('ERROR <train.py>: NO LOSS FUNCTION DEFINED!!!')
                else:
                    raise TypeError('ERROR <train.py>: unkown neural network architecture!!!')

                loss_array.append(reduced_loss)

        loss_total = tf.reduce_sum(loss_array)

        # --------------------------------------------------------------------------
        # Define evaluation metric
        # --------------------------------------------------------------------------

        evaluation_metric = []

        if self.config.ARCHITECTURE == 'semantic_segmentation':
            list_mIoU = []
            list_update_op_validation = [] 
            list_accuracy = []
            list_update_op_accuracy = []
            
            
            for iter_gpu in range(len(input_placeholder['image_batch'])):
                with tf.device('/gpu:%d' % iter_gpu):
                    if self.config.MODEL == 'SegNet_BN' or self.config.MODEL == 'SegNet_BN_encoder' or self.config.MODEL == 'SegNet_BN_decoder' or self.config.MODEL == 'SegNet_BN_encoderDecoder':
                        raw_output = net.layers['output'][iter_gpu]
                        # find index of channel with highest value
                        raw_output_up = tf.argmax(raw_output, axis=3, output_type=tf.int32)
                        raw_pred_mIoU = tf.expand_dims(raw_output_up, dim=3)
                    else:  # ICNet
                        ori_shape = self.config.DATASET_TRAIN.INPUT_SIZE #??
                        print (net.layers['output'][iter_gpu])
                        raw_output = net.layers['output'][iter_gpu]
                        raw_output_up = tf.compat.v1.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
                        raw_output_up = tf.argmax(raw_output_up, axis=3, output_type=tf.int32)
                        raw_pred_mIoU = tf.expand_dims(raw_output_up, dim=3)

                    # determine mIoU

                    if self.config.USAGE_TIMESEQUENCES:  # evaluate only last image of time sequence
                        # find indices of relevant predictions within batch
                        pred_of_interest = np.array(range(batch_size), dtype=np.int32) * self.config.TIMESEQUENCE_LENGTH + self.config.TIMESEQUENCE_LENGTH - 1
                        # Discard all but last value of each time sequence + flatten tensors
                        pred_flatten_mIoU = tf.reshape(tf.gather(raw_pred_mIoU, pred_of_interest), [-1, ])
                        raw_gt = tf.reshape(tf.gather(input_placeholder['label_batch'][iter_gpu], pred_of_interest), [-1, ])
                    else:  # evaluate all images of batch size
                        pred_flatten_mIoU = tf.reshape(raw_pred_mIoU, [-1, ])
                        raw_gt = tf.reshape(input_placeholder['label_batch'][iter_gpu], [-1, ])


                    # only use pixels for which ground-truth contains relevant classes
                    indices_mIoU = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.config.DATASET_TRAIN.NUM_CLASSES - 1)), 1)
                    gt_mIoU = tf.cast(tf.gather(raw_gt, indices_mIoU), tf.int32)
                    pred_mIoU = tf.gather(pred_flatten_mIoU, indices_mIoU)

                    mIoU, update_op_validation = tf.contrib.metrics.streaming_mean_iou(pred_mIoU, gt_mIoU, num_classes=self.config.DATASET_TRAIN.NUM_CLASSES)
                    accuracy, update_op_accuracy = tf.compat.v1.metrics.accuracy(gt_mIoU, pred_mIoU)
                    
                    list_mIoU.append(mIoU)
                    list_accuracy.append(accuracy)
                    list_update_op_validation.append(update_op_validation)
                    list_update_op_accuracy.append(update_op_accuracy)
            
        
            evaluation_metric = [mIoU, accuracy, list_update_op_validation, list_update_op_accuracy]
                
        return loss_array, evaluation_metric

    def determine_learningRate(self, step_number):
        """
        Helper function that returns learning rate tensor.
        Learning rate is determined according to policy chosen in config, using the passed step number.
        """

        if self.config.OPTIMIZER.LEARNING_RATE_POLICY == 'poly':
            base_lr = tf.constant(self.config.OPTIMIZER.LEARNING_RATE)
            learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_number / self.config.NUM_STEPS),
                                                          self.config.OPTIMIZER.POWER))
        elif self.config.OPTIMIZER.LEARNING_RATE_POLICY == 'step':
            base_lr = tf.constant(self.config.OPTIMIZER.LEARNING_RATE)
            step_decay = tf.constant(self.config.OPTIMIZER.STEP_DECAY)
            learning_rate = base_lr * tf.pow(step_decay, tf.floordiv(step_number, self.config.OPTIMIZER.STEP_SIZE))
        else:
            print("[determine_learningRate] Error: Learning-Rate-Policy not defined! Please check your spelling!")
            raise NotImplementedError
        return learning_rate

    def get_optimizer(self, step_number, reduce_learning_rate=False):
        """
        Helper function that returns optimizer operation as chosen in config.
        Step number is used to calculate learning rate.
        """

        if self.config.OPTIMIZER.TYPE == 'MomentumOptimizer':
            learning_rate = self.determine_learningRate(step_number)
            if reduce_learning_rate:
                learning_rate *= self.config.LEARNING_RATE_REDUCTION_FACTOR
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, self.config.OPTIMIZER.MOMENTUM)
        elif self.config.OPTIMIZER.TYPE == 'AdamOptimizer':
            learning_rate = self.determine_learningRate(step_number)  # Use learning rate policy for Adam, since this
            # proved to be superior in tests (even though theoretically it should not be needed)
            if reduce_learning_rate:
                learning_rate *= self.config.LEARNING_RATE_REDUCTION_FACTOR
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.config.OPTIMIZER.TYPE == 'GradientDescentOptimizer':
            learning_rate = self.determine_learningRate(step_number)
            if reduce_learning_rate:
                learning_rate *= self.config.LEARNING_RATE_REDUCTION_FACTOR
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        else:
            print("[get_optimizer] Error: Optimizer not defined! Please check your spelling!")

        return optimizer

    def create_accumulating_training_op(self, optimizer, grads_and_vars, global_step=None, average_grads=True):
        """
        Set up an operation that accumulates gradients for self.config.ITER_SIZE executions and only actually applies
        (the average of) the accumulated gradients to the variables every ITER_SIZE-th execution.

        :param optimizer: The tf.train.Optimizer that should be used to apply the gradients
        :param grads_and_vars: List of (gradient, variable) pairs as returned by optimizer.compute_gradients()
        :param global_step: Optional tf.Variable to increment by one after each execution of the returned operation
        :param average_grads: Whether the accumulated gradients should be averaged before applying them. If the used
                              loss averages over the samples in a batch, this should usually be True
        :return: An operation that accumulates gradients for self.config.ITER_SIZE iterations before applying them
        """
        # Loosely based on
        # https://github.com/tensorpack/tensorpack/blob/45ebac959f34507f29176fc12d327f3cc9ff7468/tensorpack/tfutils/optimizer.py#L133-L202
        # and
        # https://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients

        # If Iter_Size is 1, simply use normal minimization
        if self.config.ITER_SIZE == 1:
            return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Create variables in which gradients are accumulated and a step counter
        # (Without any dependencies, as they would be triggered during initialization due to the call tf.zeros_like())
        with tf.control_dependencies(None), tf.compat.v1.variable_scope("AccumGradOptimizer"):
            trainable_vars = [gv_pair[1] for gv_pair in grads_and_vars]
            accumulators = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False, name="accumulator")
                            for var in trainable_vars]  # use tf.Variable() instead of tf.get_variable(), since that
            # method uses the (uniquified) name scope as prefix (instead of non-uniquified variable scope) and
            # automatically uniquifies variable names
            accumulation_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="counter")

        with tf.name_scope("AccumGradOptimizer"):
            # Create an operation which sums up gradients and increments counter
            ops = [accumulator.assign_add(grad) for (accumulator, (grad, var)) in zip(accumulators, grads_and_vars)]
            ops.append(accumulation_counter.assign_add(1))
            if global_step is not None:
                ops.append(global_step.assign_add(1))
            accumulate_op = tf.group(*ops, name="accumulate_grads")

            # Define a function, in which an operation is created and returned that applies the accumulated gradients to
            # the trainable variables and resets the accumulation variables afterwards.
            # Nesting this in a function is needed for the condition operation following later
            def apply_grads_fn():
                if average_grads:
                    iter_size = tf.convert_to_tensor(self.config.ITER_SIZE, dtype=tf.float32, name="iter_size")
                    apply_grads_op = optimizer.apply_gradients([(accumulator / iter_size, var)
                                                                for (accumulator, (grad, var))
                                                                in zip(accumulators, grads_and_vars)])
                else:
                    apply_grads_op = optimizer.apply_gradients([(accumulator, var)
                                                                for (accumulator, (grad, var))
                                                                in zip(accumulators, grads_and_vars)])

                # Make sure that applying the gradients is always executed before zeroing the accumulation variables
                with tf.control_dependencies([apply_grads_op]):
                    zero_ops = [accumulator.assign(tf.zeros_like(accumulator)) for accumulator in accumulators]
                    zero_ops.append(accumulation_counter.assign(0))

                return tf.group(*zero_ops, name="apply_grads")

            # Make sure that the accumulation is executed every time train_op is run
            with tf.control_dependencies([accumulate_op]):
                # Apply gradients every time the counter reaches ITER_SIZE, otherwise don't execute any further ops
                reached_accum_end = tf.equal(tf.mod(accumulation_counter, self.config.ITER_SIZE), 0)
                train_op = tf.cond(reached_accum_end, apply_grads_fn, tf.no_op).op

        return train_op

    def evaluateResult(self, sess, mIoU, accuracy, update_op_validation, update_op_accuracy, input_placeholder,
                       dataset):
        """
        Helper function that calculates evaluation metrics over entire data set

        :param sess: TF-Session in which calculation should be run
        :param mIoU: Tensor containing mIoU
        :param accuracy: Tensor containing accuracy
        :param update_op_validation: TF-OP to update mIoU value
        :param update_op_accuracy: TF-OP to update accuracy value
        :param input_placeholder: Dictionary of input placeholders of network
        :param dataset: Choose 'val' or 'train' data set
        :return: mIoU and accuracy for semantic segmentation, meanAP and accuracy for object detection
        """

        sess.run(tf.compat.v1.local_variables_initializer())  # Local vars are accuracy/total, accuracy/count and
        # mean_iou/total_confusion_matrix
        if dataset == 'val':
            imagereader_val = ImageReader(self.config.IMAGEREADER.VAL, self.config.DATASET_VAL, self.config.BATCH_SIZE, self.config.TIMESEQUENCE_LENGTH)
            text = 'evaluation - validation set'
            #CLASSES = self.config.DATASET_VAL.CLASSES
        else:
            imagereader_val = ImageReader(self.config.IMAGEREADER.VAL, self.config.DATASET_TRAIN, self.config.BATCH_SIZE, self.config.TIMESEQUENCE_LENGTH)
            text = 'evaluation - training set'
            #CLASSES = self.config.DATASET_TRAIN.CLASSES

        if self.config.ARCHITECTURE == 'semantic_segmentation':
            return self.evaluateResult_semanticSegmentation(sess, mIoU, accuracy, update_op_validation,
                                                            update_op_accuracy, input_placeholder, text,
                                                            imagereader_val)
        else:
            print('[ERROR] network architecture does not exist!!! Please check your spelling!')
            raise NotImplementedError

    def evaluateResult_semanticSegmentation(self, sess, mIoU, accuracy, update_op_validation, update_op_accuracy,
                                            input_placeholder, text, imagereader_val):
        """Helper function that calculates mIoU and accuracy over entire data set used by passed imagereader"""

        # Iterate over each image of training set
        for iter in trange(int(imagereader_val._dataset_amount/(self.config.BATCH_SIZE*self.config.TIMESEQUENCE_LENGTH)),
                           desc=text, leave=True):

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

            _ = sess.run([update_op_validation, update_op_accuracy], feed_dict=feed_dict)

        mIoU_value = sess.run(mIoU)
        accuracy_value = sess.run(accuracy)
        sess.run(tf.compat.v1.local_variables_initializer())  # Local vars are accuracy/total, accuracy/count and
        # mean_iou/total_confusion_matrix

        return mIoU_value, accuracy_value

    

    #####################################################################################################
    ###                                                                                               ###
    ###                                          Training                                             ###
    ###                                                                                               ###
    #####################################################################################################

    def do_training(self, train_op, update_op_validation, update_op_accuracy, loss, mIoU, accuracy, input_placeholder):
        """
        Helper function to start session, load pretrained model and run the entire training session (by iterating over
        data set for config.NUM_STEPS, adjusting weights (by running train_op) and calculating loss in each step).

        If requested in config, evaluate performance before and during training.
        """

        # ----------------------------------------------
        # Set up tf session and initialize variables
        # ----------------------------------------------

        sessConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sessConfig.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=sessConfig)

        sess.run(tf.compat.v1.global_variables_initializer())


        # -------------------------
        # Load pretrained model
        # -------------------------

        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=1)

        if self.config.USE_PRETRAINED_MODEL:
            print('use pretrained model')

            if (self.config.PRETRAINED_MODEL.endswith('.npy')):

                BN_param_map = {'scale':    'gamma',
                    'offset':   'beta',
                    'variance': 'moving_variance',
                    'mean':     'moving_mean'}

                data_dict = np.load(self.config.PRETRAINED_MODEL, encoding='latin1', allow_pickle=True).item()

                for op_name in data_dict:
                    with tf.compat.v1.variable_scope(op_name, reuse=True):
                        ignore_missing = True
                        for param_name, data in data_dict[op_name].items():
                            try:
                                if 'bn' in op_name:
                                    param_name = BN_param_map[param_name]

                                var = tf.compat.v1.get_variable(param_name)
                                sess.run(var.assign(data))
                            except ValueError:
                                if not ignore_missing:
                                    raise
            else:
                ckpt = tf.compat.v1.train.get_checkpoint_state(self.config.PRETRAINED_MODEL)
                if ckpt and ckpt.model_checkpoint_path:

                    # get variables, which are stored in checkpoint-file
                    from tensorflow.contrib.framework.python.framework import checkpoint_utils
                    vars_to_check = tf.compat.v1.global_variables()

                    # remove optimizer variables from vars_to_check
                    vars_to_check = [var for var in vars_to_check if (('Momentum' not in var.name)
                                                                        and ('global_step' not in var.name)
                                                                        and ('Adam' not in var.name)
                                                                        and ('beta1_power' not in var.name)
                                                                        and ('beta2_power' not in var.name)
                                                                        and ('AccumGradOptimizer' not in var.name))]

                    check_var_list = checkpoint_utils.list_variables(self.config.PRETRAINED_MODEL)
                    check_var_list = [x[0] for x in check_var_list]
                    check_var_set = set(check_var_list)
                    vars_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] in check_var_set]
                    vars_not_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] not in check_var_set]

                    # load variables from checkpoint
                    if len(vars_in_checkpoint):
                        loader = tf.compat.v1.train.Saver(var_list=vars_in_checkpoint)
                        self.load(loader, sess, ckpt.model_checkpoint_path)
                    else:
                        print ("[WARNING] no variables to load from checkpoint found!")
                else:
                    print('cannot find model')

        # -----------------------------------
        # Initialize evaluation parameters
        # -----------------------------------

        best_mIoU = 0
        best_epoch = 0
        duration_evals_total = 0
        current_eval_count = 0
        training_monitoring = np.zeros((int(self.config.NUM_STEPS / self.config.SAVE_PRED_EVERY) + 2, 6))
        max_num_evals = int(math.ceil(float(self.config.MAX_ITERATIONS)/float(self.config.SAVE_PRED_EVERY)))

        # Initialize training monitoring

        if self.config.ENABLE_VISUAL_MONITORING:
            init_trainingMonitoring(training_monitoring, self.config.NUM_STEPS)

        # Do evaluation before training if requested
        if self.config.EVALUATION.BEFORETRAINING:

            startTime_eval = time.time()

            # Evaluate on training set

            if self.config.EVALUATION.DOEVALUATION_TRAIN:
                mIoU_train, acc_train = self.evaluateResult(sess, mIoU, accuracy, update_op_validation,
                                                            update_op_accuracy, input_placeholder, 'train')
            else:
                mIoU_train = 0.0
                acc_train = 0.0
            print('accuracy-train: {}'.format(acc_train))
            print('mIoU-train: {}'.format(mIoU_train))

            # Evaluate on validation set

            mIoU_eval, acc_eval = self.evaluateResult(sess, mIoU, accuracy, update_op_validation, update_op_accuracy,
                                                      input_placeholder, 'val')
            print('accuracy-val: {}'.format(acc_eval))
            print('mIoU-val: {}'.format(mIoU_eval))

            # Determine difference between training and evaluation metric values

            mIoU_diff = mIoU_train - mIoU_eval
            print('mIoU diff: {}'.format(mIoU_diff))

            # Write results to training_monitoring
            training_monitoring[0] = [0, mIoU_train, mIoU_eval, mIoU_diff, acc_train, acc_eval]
            best_mIoU = mIoU_eval
            best_epoch = 0

            duration_eval = time.time() - startTime_eval
            duration_evals_total += duration_eval
            current_eval_count += 1
            max_num_evals += 1  # increase by one, because calculation of value does not include optional evaluation
            # before training

            # Update visual training monitoring
            if self.config.ENABLE_VISUAL_MONITORING:
                update_TrainingMonitoring(training_monitoring, 0)

        # ----------------------------
        # Run training steps
        # ----------------------------

        # Save initial checkpoint
        self.save(saver, sess, 0)

        # Initialize ImageReader
        imagereader_train = ImageReader(self.config.IMAGEREADER.TRAIN, self.config.DATASET_TRAIN, self.config.BATCH_SIZE, self.config.TIMESEQUENCE_LENGTH)

        # Iterate over training steps
        print("Total number of steps: {}".format(self.config.NUM_STEPS))
        print("self.config.SAVE_PRED_EVERY = {}".format(self.config.SAVE_PRED_EVERY))
        duration_trainsteps_total = 0
        for step in range(self.config.NUM_STEPS):
            start_time = time.time()

            # Get next mini-batch and feed it into network placeholders
            
            feed_dict_train = {input_placeholder['step_ph']: step,
                               input_placeholder['is_training']: True}

            training_batch = imagereader_train.getNextMinibatch()

            for iter_GPU in range(len(input_placeholder['image_batch'])):        
                num_GPU = len(input_placeholder['image_batch'])
                batch_size = training_batch['blob_data'].shape[0]
                batch_size_local = batch_size // num_GPU
                if (iter_GPU == (num_GPU -1)):
                    batch_size_act = batch_size - (num_GPU - 1) * (batch_size // num_GPU)
                else:
                    batch_size_act = batch_size // num_GPU

                feed_dict_train.update({input_placeholder['image_batch'][iter_GPU]: training_batch['blob_data'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:],
                                        input_placeholder['label_batch'][iter_GPU]: training_batch['blob_label'][iter_GPU*batch_size_local:iter_GPU*batch_size_local+batch_size_act,:,:,:]})
                               
            

            # Run one training step (calculates loss and adjusts trainable variables)
            #_,_, loss_value = sess.run((train_op, update_ops, loss), feed_dict=feed_dict_train)
            _, loss_value = sess.run((train_op, loss), feed_dict=feed_dict_train)

            # Calculate duration of training steps
            # (Exclude first step from calculation of average as it always takes a lot longer than the other steps)
            duration_trainstep = time.time() - start_time
            duration_trainsteps_total += duration_trainstep if not step == 0 else 0
            duration_trainsteps_avg = duration_trainsteps_total/step if not step == 0 else 0
            duration_eval_avg = duration_evals_total/current_eval_count if not current_eval_count == 0 else 0

            # Print performance of this step, time taken and estimated time left
            print('step {:d} - total loss = {:.3f} ({:.3f} sec for this step, on average {:.3f} sec/step) '
                  '\t remaining training time: {:.3f}h'
                  .format(step, loss_value, duration_trainstep, duration_trainsteps_avg,
                          ((self.config.NUM_STEPS - step - 1) * duration_trainsteps_avg
                           + (max_num_evals - current_eval_count) * duration_eval_avg)/3600.0))

            # ----------------------------------------------------------------------
            # Evaluate during training (if requested) and save model parameters
            # ----------------------------------------------------------------------

            if ((step + 1) % self.config.SAVE_PRED_EVERY == 0) or ((step + 1) == self.config.NUM_STEPS):
                if self.config.EVALUATION.DOEVALUATION:
                    sys.stdout.flush()

                    startTime_eval = time.time()

                    # Evaluate on training set

                    if self.config.EVALUATION.DOEVALUATION_TRAIN:
                        mIoU_train, acc_train = self.evaluateResult(sess, mIoU, accuracy, update_op_validation,
                                                                    update_op_accuracy, input_placeholder, 'train')
                    else:
                        mIoU_train = 0.0
                        acc_train = 0.0

                    print ('accuracy-train: {}'.format(acc_train))
                    print ('mIoU-train: {}'.format(mIoU_train))

                    # Evaluate on validation set

                    mIoU_eval, acc_eval = self.evaluateResult(sess, mIoU, accuracy, update_op_validation,
                                                              update_op_accuracy, input_placeholder, 'val')
                    print ('accuracy-val: {}'.format(acc_eval))
                    print ('mIoU-val: {}'.format(mIoU_eval))

                    # Determine difference between training and evaluation metric values

                    mIoU_diff = mIoU_train - mIoU_eval
                    print('mIoU diff: {}'.format(mIoU_diff))

                    # -----------------------------
                    # for training monitoring
                    # -----------------------------

                    duration_eval = time.time() - startTime_eval
                    duration_evals_total += duration_eval
                    current_eval_count += 1

                    # Write results to training_monitoring
                    iter = int((step + 1) / self.config.SAVE_PRED_EVERY)
                    training_monitoring[iter] = [step + 1, mIoU_train, mIoU_eval, mIoU_diff, acc_train, acc_eval]

                    # Save best results (including saving weights to config.SNAPSHOT_DIR)
                    if best_mIoU < mIoU_eval:
                        best_mIoU = mIoU_eval
                        best_epoch = iter
                        self.save(saver, sess, step + 1)

                    # Update visual training monitoring
                    if self.config.ENABLE_VISUAL_MONITORING:
                        update_TrainingMonitoring(training_monitoring, iter, best_epoch=best_epoch)

                else:  # no evaluation should be done
                    # simply save newest parameters
                    self.save(saver, sess, step + 1)
                    print('Ready')

        # --------------------------------
        # Final tasks at end of training
        # --------------------------------

        # Close session

        sess.close()
        tf.compat.v1.reset_default_graph()

        # Write training parameters/monitoring to file
        saveTrainingConfig(training_monitoring, best_epoch, self.config)

        # Keep showing Training Monitoring
        if self.config.ENABLE_VISUAL_MONITORING:
            hold_plot()

    def serial_training(self):
        """
        Function for training in a single device.
        Set up placeholders and network graph, define loss, optimizer and evaluation metrics, then execute training
        session.
        """

        # Define input placeholders

        input_placeholder = {}
        input_placeholder.update({'step_ph': tf.compat.v1.placeholder(dtype=tf.float32, shape=())})  # actual training iteration
                                                                                 # --> used for adapting learning rate
        input_placeholder.update({'is_training': tf.compat.v1.placeholder(dtype=tf.bool, shape=())})

        if self.config.ARCHITECTURE == 'semantic_segmentation':
            batch_size = self.config.BATCH_SIZE * self.config.TIMESEQUENCE_LENGTH

            # Search for available GPUs: the result is a list of device ids like `['/gpu:0', '/gpu:1']`
            devices = self.get_available_gpus()
            print ("found devices: ", devices)
            num_GPU = len(devices)
            if (num_GPU) == 0:
                num_GPU = 1 # CPU support!
            # min 1 sample should be applied on a GPU
            if (self.config.BATCH_SIZE < num_GPU):
                num_GPU = self.config.BATCH_SIZE

            image_placeholder = []
            label_placeholder = []
            for iter_GPU in range(num_GPU):
                if (iter_GPU == (num_GPU -1)):
                    batch_size_local = batch_size - (num_GPU - 1) * (batch_size // num_GPU)
                else:
                    batch_size_local = batch_size // num_GPU 
                print ('batch_size /gpu:{} : {}'.format(iter_GPU, batch_size_local))
   
                image_placeholder.append(tf.compat.v1.placeholder(dtype=tf.float32,
                                                                    shape=(batch_size_local,
                                                                           self.config.DATASET_TRAIN.INPUT_SIZE[0],
                                                                           self.config.DATASET_TRAIN.INPUT_SIZE[1],
                                                                           self.config.DATASET_TRAIN.NUM_CHANNELS)))
                label_placeholder.append(tf.compat.v1.placeholder(dtype=tf.float32,
                                                                    shape=(batch_size_local,
                                                                           self.config.DATASET_TRAIN.INPUT_SIZE[0],
                                                                           self.config.DATASET_TRAIN.INPUT_SIZE[1],
                                                                           1)))

            input_placeholder.update({'image_batch': image_placeholder})
            input_placeholder.update({'label_batch': label_placeholder})
        else:
            print('[ERROR] network architecture does not exist!!! Please check your spelling!')
            raise NotImplementedError

        # Get optimizer operation and set up network
        optimizer = self.get_optimizer(input_placeholder['step_ph'])
        loss_array, evaluation_metric = self.network(input_placeholder)

        # Get list of trainable variables. Train beta and gamma of batch normalisation only if requested in config.
        all_trainable = [v for v in tf.compat.v1.trainable_variables()
                         if (('beta' not in v.name and 'gamma' not in v.name) or self.config.TRAIN_BETA_GAMMA)]
        if self.config.PRINT_TRAINABLE_PARAMETERS:
            for ix in range(len(all_trainable)):
                print (all_trainable[ix])


        # Get training operation for adjusting trainable variables
        #global_step = tf.compat.v1.train.get_or_create_global_step()

        # compute gradients on each GPU
        tower_grads = []
        for iter_gpu in range(len(loss_array)):
            with tf.device('/gpu:%d' % iter_gpu):
                with tf.name_scope("compute_gradients"):
                    grads_and_vars = optimizer.compute_gradients(loss_array[iter_gpu], all_trainable)
                    tower_grads.append(grads_and_vars)

        # average gradients
        
        controller="/cpu:0"
        # Apply the gradients on the controlling device (= parameter server)
        with tf.name_scope("apply_gradients"), tf.device(controller):
            # Note that what we are doing here mathematically is equivalent to returning the
            # average loss over the towers and compute the gradients relative to that.
            # Unfortunately, this would place all gradient-computations on one device, which is
            # why we had to compute the gradients above per tower and need to average them here.

            if not self.config.USE_GRADIENT_CLIPPING:
                grads_and_vars = self.average_gradients(tower_grads)
            else:
                grads, vars = zip(*self.average_gradients(tower_grads))
                grads, _ = tf.clip_by_global_norm(grads, self.config.MAX_GRAD_NORM)
                grads_and_vars = zip(grads, vars)

            global_step = tf.compat.v1.train.get_or_create_global_step()
            train_op = self.create_accumulating_training_op(optimizer, grads_and_vars, global_step=global_step)

            # Get moving_mean and moving_variance update operations from tf.compat.v1.GraphKeys.UPDATE_OPS
            if self.config.UPDATE_MEAN_VAR:
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                train_op = tf.group(train_op, update_ops)

            loss = tf.reduce_mean(loss_array)

        # Define evaluation metrics

        update_op_validation = []
        update_op_accuracy = []
        mIoU = []
        accuracy = []

        if self.config.ARCHITECTURE == 'semantic_segmentation':
            mIoU = evaluation_metric[0]
            accuracy = evaluation_metric[1]
            update_op_validation = evaluation_metric[2]
            update_op_accuracy = evaluation_metric[3]

        # Run training session
        self.do_training(train_op, update_op_validation, update_op_accuracy, loss, mIoU, accuracy, input_placeholder)


def main():
    start_time = time.time()

    # get arguments
    args = get_arguments()

    # init config
    config = init_local_config(args)

    # adjust config for training
    config = adjust_config(config)

    # start training
    training = Training(config)

    total_duration = time.time() - start_time
    print("The execution of this script took {:.3f}h in total.".format(total_duration/3600.0))


if __name__ == '__main__':
    main()
