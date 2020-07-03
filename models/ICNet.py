# original network of the implementation https://github.com/hellochick/ICNet-tensorflow


from network import Network
from loss import *
import tensorflow as tf

class ICNet(Network):
    def setup(self, is_training, num_classes, evaluation, timeSequence=1):
        (self.feed('data')
             .interp(s_factor=0.5, name='data_sub2')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_1_3x3_s2')
             .conv(3, 3, 32, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_2_3x3')
             .conv(3, 3, 64, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_3_3x3')
             .zero_padding(paddings=1, name='padding0')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_proj'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_1_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_increase'))

        (self.feed('conv2_1_1x1_proj',
                   'conv2_1_1x1_increase')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_2_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_2_1x1_increase'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_3_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_3_1x1_increase'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 256, 2, 2, biased=True, relu=False, name='conv3_1_1x1_proj'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 64, 2, 2, biased=True, relu=True, name='conv3_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_1_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_1_1x1_increase'))

        (self.feed('conv3_1_1x1_proj',
                   'conv3_1_1x1_increase')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .interp(s_factor=0.5, name='conv3_1_sub4')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_2_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_2_1x1_increase'))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_3_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_3_1x1_increase'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_4_1x1_reduce')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_4_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_4_1x1_increase'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_proj'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_1_1x1_reduce')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_1_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_increase'))

        (self.feed('conv4_1_1x1_proj',
                   'conv4_1_1x1_increase')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_2_1x1_reduce')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_2_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_2_1x1_increase'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_3_1x1_reduce')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_3_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_3_1x1_increase'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_4_1x1_reduce')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_4_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_4_1x1_increase'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_5_1x1_reduce')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_5_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_5_1x1_increase'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_6_1x1_reduce')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_6_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_6_1x1_increase'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_proj'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_1_1x1_reduce')
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_1_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_increase'))

        (self.feed('conv5_1_1x1_proj',
                   'conv5_1_1x1_increase')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_2_1x1_reduce')
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_2_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_2_1x1_increase'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_3_1x1_reduce')
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_3_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_3_1x1_increase'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

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
            .avg_pool(h/6, w/6, h/6, w/6, name='conv5_3_pool6')
            .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_4_k1')
             .interp(z_factor=2.0, name='conv5_4_interp')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub4'))

        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_1_sub2_proj'))

        (self.feed('conv_sub4',
                   'conv3_1_sub2_proj')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(z_factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub2'))

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_sub1')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv2_sub1')
             .conv(3, 3, 64, 2, 2, biased=True, padding='SAME', relu=True, name='conv3_sub1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_sub1_proj'))

        (self.feed('conv_sub2',
                   'conv3_sub1_proj')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(z_factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls'))

        # Get class scores from CFFs for cascade label guidance
        (self.feed('conv5_4_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub4_out'))
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


