

from network import Network
from loss import *
import tensorflow as tf



####################################################################################################################################
###                                                                                                                              ###
### Version 6: add a convLSTM at the end of each of the three branches and before the softmax layer                              ###
###                                                                                                                              ###
####################################################################################################################################


class LSTM_ICNet_v6(Network):

    VARIANTS = ["end2end",
                "freezeBranches",
                "spatial_end2end",
                "spatial_freezeBranches",
                "depthwise_end2end",
                "depthwise_freezeBranches",
                "separable_end2end",
                "separable_freezeBranches"]

    def setup(self, is_training, num_classes, evaluation, timeSequence=1, variant=None):

        if isinstance(variant, str):
            if variant not in self.VARIANTS:
                raise ValueError("Unknown model variant was chosen. Please check your spelling")
        elif isinstance(variant, int):
            variant = self.VARIANTS[variant]
        elif variant is None:
            variant = self.VARIANTS[0]  # default to end2end (standard training and convLSTMs of version 2)
            print("Using default variant as None was passed to variant selection!")
        else:
            raise TypeError("Passed model variant must be string or int!")

        # whether to freeze branches of not during training

        if variant == "freezeBranches" or variant == "spatial_freezeBranches" or variant == "depthwise_freezeBranches" or variant == "separable_freezeBranches":
            freeze_branches = True
        else:
            freeze_branches = False

        # select convLSTM-type

        if variant == "spatial_end2end" or variant == "spatial_freezeBranches":
            convLSTM_type = "spatial"
        elif variant == "depthwise_end2end" or variant == "depthwise_freezeBranches":
            convLSTM_type = "depthwise"
        elif variant == "separable_end2end" or variant == "separable_freezeBranches":
            convLSTM_type = "separable"
        else:
            convLSTM_type = "convolution" 
    
        # Start of low and medium resolution branches

        (self.feed('data')
             .interp(s_factor=0.5, name='data_sub2')
             .conv(3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name='conv1_1_3x3_s2', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv1_1_3x3_s2_bn', not_trainable=freeze_branches)
             .conv(3, 3, 32, 1, 1, biased=False, padding='SAME', relu=False, name='conv1_2_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv1_2_3x3_bn', not_trainable=freeze_branches)
             .conv(3, 3, 64, 1, 1, biased=False, padding='SAME', relu=False, name='conv1_3_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv1_3_3x3_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding0')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn', not_trainable=freeze_branches))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_1_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_1_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_2_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_2_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_3_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_3_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn', not_trainable=freeze_branches))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 64, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_1_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_1_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')  # this is the end of the medium resolution branch
             .interp(s_factor=0.5, name='conv3_1_sub4')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_2_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_2_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_3_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_3_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_4_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_4_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn', not_trainable=freeze_branches))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_1_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_1_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_2_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_2_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_3_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_3_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_4_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_4_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_5_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_5_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_6_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv4_6_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn', not_trainable=freeze_branches))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_1_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_1_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_2_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_2_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn', not_trainable=freeze_branches)
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_3_3x3', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_3_3x3_bn', not_trainable=freeze_branches)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase', not_trainable=freeze_branches)
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn', not_trainable=freeze_branches))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        # Define Pyramid Pooling

        shape = self.layers['conv5_3/relu'][0].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_3/relu')
             .avg_pool(h, w, h, w, name='conv5_3_pool1')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/2, w/2, h/2, w/2, name='conv5_3_pool2')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/3, w/3, h/3, w/3, name='conv5_3_pool3')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/4, w/4, h/4, w/4, name='conv5_3_pool6')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        # Combine parallel pooling stages and input feature map
        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_4_k1', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv5_4_k1_bn', not_trainable=freeze_branches)
             .interp(z_factor=2.0, name='conv5_4_interp')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv_sub4')
              # add convLSTm here
	         .convlstm(timeSequence,3,3,128, name='convlstm5_4_sub4', convtype=convLSTM_type)
              # add end
             .batch_normalization(relu=False, name='conv_sub4_bn'))

        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_1_sub2_proj')
             # add convLSTM here
             .convlstm(timeSequence,3,3,128, name='convlstm3_1_sub2', convtype=convLSTM_type)
             # add end
             .batch_normalization(relu=False, name='conv3_1_sub2_proj_bn'))

        (self.feed('conv_sub4_bn',
                   'conv3_1_sub2_proj_bn')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(z_factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv_sub2')
             .batch_normalization(relu=False, name='conv_sub2_bn'))

        # Pfad 3

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name='conv1_sub1', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv1_sub1_bn', not_trainable=freeze_branches)
             .conv(3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name='conv2_sub1', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv2_sub1_bn', not_trainable=freeze_branches)
             .conv(3, 3, 64, 2, 2, biased=False, padding='SAME', relu=False, name='conv3_sub1', not_trainable=freeze_branches)
             .batch_normalization(relu=True, name='conv3_sub1_bn', not_trainable=freeze_branches)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_sub1_proj')
              # add convLSTm here
	         .convlstm(timeSequence,3,3,128, name='convlstm3_sub1', convtype=convLSTM_type)
              # add end
             .batch_normalization(relu=False, name='conv3_sub1_proj_bn'))

        # Upsampling 

        (self.feed('conv_sub2_bn',
                   'conv3_sub1_proj_bn')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             # add convLSTm here
	         .convlstm(timeSequence,3,3,128, name='convlstm6_cls', convtype=convLSTM_type)
             .batch_normalization(relu=True, name='convlstm6_cls_bn')   
             # add end
             .interp(z_factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls'))

        # Get class scores from CFFs for cascade label guidance
        (self.feed('conv5_4_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub4_out', not_trainable=freeze_branches))
        (self.feed('sub24_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub24_out'))

        # map variable to output
        (self.feed('conv6_cls').identity(name="output"))






    def get_trainingLoss(self, label_batch, config, iter_gpu = 0):

        if config.LOSS.TYPE == 'softmax_cross_entropy':

            # Get corrsponding outputs  
 
            sub4_out = self.layers['sub4_out'][iter_gpu]
            sub24_out = self.layers['sub24_out'][iter_gpu]
            sub124_out = self.layers['output'][iter_gpu]

            # evaluate only last image of image sequence --> fetch this last image
            if config.USAGE_TIMESEQUENCES:
                # if batchsize = 1: use only last image of batch and expand dimension
                if config.BATCH_SIZE == 1:
                    label_batch = tf.expand_dims(label_batch[-1, ...], 0)  # discard all but last labeled image in batch
                    sub4_out = tf.expand_dims(sub4_out[-1, ...], 0)  # discard all but last output image in batch
                    sub24_out = tf.expand_dims(sub24_out[-1, ...], 0)
                    sub124_out = tf.expand_dims(sub124_out[-1, ...], 0)
                # if batchsize > 1: use only every n-th image, where n = length of sequence
                else:
                    pred_of_interest = np.array(range(batch_size), dtype=np.int32) * config.TIMESEQUENCE_LENGTH + config.TIMESEQUENCE_LENGTH - 1
                    label_batch = tf.gather(label_batch, pred_of_interest)
                    sub4_out = tf.gather(sub4_out, pred_of_interest)
                    sub24_out = tf.gather(sub24_out, pred_of_interest)
                    sub124_out = tf.gather(sub124_out, pred_of_interest)

            # determine loss 
            loss_sub4 = createLoss_softmaxCrossEntropy(sub4_out, label_batch, config.DATASET_TRAIN.NUM_CLASSES, config.DATASET_TRAIN.IGNORE_LABEL)
            loss_sub24 = createLoss_softmaxCrossEntropy(sub24_out, label_batch, config.DATASET_TRAIN.NUM_CLASSES, config.DATASET_TRAIN.IGNORE_LABEL)
            loss_sub124 = createLoss_softmaxCrossEntropy(sub124_out, label_batch, config.DATASET_TRAIN.NUM_CLASSES, config.DATASET_TRAIN.IGNORE_LABEL)
            l2_losses = [config.OPTIMIZER.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if 'weights' in v.name or 'kernel' in v.name]

            # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
            total_loss = config.LOSS.LAMBDA1 * loss_sub4 +  config.LOSS.LAMBDA2 * loss_sub24 + config.LOSS.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)
            #total_loss = tf.Print(total_loss, [total_loss], "total_loss: "+str(iter_gpu), summarize=10, first_n=-1)
        else:
            raise TypeError("Loss Function is not defined. Please check your spelling!")

        return total_loss




