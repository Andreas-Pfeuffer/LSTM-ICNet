import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow as tf

label_colours_19 = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle


label_colours_14 = [[0,0,0], [69, 69, 69], [0, 0, 142],
                # 0 = Misc , 1 = Building, 2 = Car
                [0, 79, 100], [153, 153, 153], [128, 64, 128], 
                # 3 = GuardRAil, 4 = Pole, 5 = Road
                [69, 129, 180], [152, 250, 152], [250, 170, 29],
                # 6 = Sky       7 = Terrain      8 = TrafficLight
                [219, 219, 0],  [0, 255, 0]  , [0, 0, 69],
                # 9 = TrafficSign 10 = Tree 11=Truck
                [0, 60, 100],  [106, 142, 35]]
                # 12 = Van 13 = Vegetation


label_colours_12 = [[0,0,0], [170,	85,	255], [244, 35, 232]
                # 0 = unlabeled, 1 = road, 2 = sidewalk,
                ,[152,	251,	152], [250,	170,	30], [220,	220,	0]
                # 3 = pole, 4 = traffic light, 5 = traffic sign
                ,[220,	20,	60], [155,	40,	0], [0,	0,	255]
                # 6 = person   7 = bicycle,     8 = car
                ,[0,	255,	255], [69, 129, 180], [0,	0,	130]
                # 9 = truck      , 10 = bus,    11 = motorcycle
                ]


def decode_labels(mask, img_shape, num_classes):
    if num_classes == 12:
        color_table = label_colours_12
    elif num_classes == 14:
        color_table = label_colours_14
    elif num_classes == 19:
        color_table = label_colours_19
    else:
        print ('ERROR<tools.py> No valid label color available!!!')

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
    
    return pred

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.compat.v1.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch
