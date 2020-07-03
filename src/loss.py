# coding=utf-8

import tensorflow as tf
import numpy as np


def get_mask(gt, num_classes, ignore_label):  
    """Get indices at which relevant classes are located within ground-truth"""

    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices


def createLoss_softmaxCrossEntropy(output, label, num_classes, ignore_label):  
    """Define softmax cross-entropy loss"""
    from tools import decode_labels, prepare_label

    raw_pred = tf.reshape(output, [-1, num_classes])  # force 2nd dimension to be of length num_classes
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])  # flatten tensor

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss




def createLoss_crossEntropyMedianFrequencyBalancing(output, label, num_classes, ignore_label, loss_weight):
    """Define cross-entropy loss with median frequency balancing"""

    raw_pred = tf.reshape(output, [-1, num_classes])  # force 2nd dimension to be of length num_classes
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])  # flatten tensor

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)
    gt = tf.one_hot(gt, depth=num_classes)

    loss = tf.nn.weighted_cross_entropy_with_logits(targets=gt, logits=pred, pos_weight=loss_weight)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss




