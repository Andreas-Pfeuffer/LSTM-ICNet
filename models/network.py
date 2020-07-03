import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib import rnn
from operations import *



DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'
layer_name = []
BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}

controller="/cpu:0"
# code based on https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
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
                
def layer(op):
    '''Decorator for composable network layers.'''
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = [self.terminals[0]]
        else:
            layer_input = []
            for i in range(len(self.terminals)):
                layer_input.append(self.terminals[i])
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        layer_name.append(name)
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, num_classes, filter_scale, evaluation=False, is_inference=False, trainable=True,
                 is_training=False, timeSequence=1, variant=None):
        """
        Construct network graph

        :param inputs: Network inputs as dictionary
        :param num_classes: Amount of output classes
        :param filter_scale: Factor by which the default amount of channels of each internal layer is scaled
        :param evaluation: If true, network is configured for evaluation
        :param is_inference: if true, network is configured for inference (tf.cond are removed for inference)
        :param trainable: Whether newly created variables should be set as trainable
        :param is_training: Switches between training mode and inference, e.g. for batch norm
        :param timeSequence: Length of time sequences
        :param variant: Variant of network to use (if variants are available)
        """

        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.is_training = is_training
        self.trainable = trainable

        # Switch variable for dropout
        self.use_dropout = tf.compat.v1.placeholder_with_default(tf.constant(1.0),
                                                                shape=[],
                                                                name='use_dropout')
        self.evaluation = evaluation
        self.is_inference = is_inference
        self.filter_scale = filter_scale

        if variant is None:
            self.setup(is_training, num_classes, evaluation, timeSequence)
        else:
            self.setup(is_training, num_classes, evaluation, timeSequence, variant)


    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    """
    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.compat.v1.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]

                        var = tf.compat.v1.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise
    """

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_shape(self, *args):
        assert len(args) != 0
        self.shapes = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer_shape = self.layers[fed_layer][0].shape
                    return fed_layer_shape
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.shapes.append(fed_layer_shape)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, not_trainable=False):
        """Creates a new TensorFlow variable. If not_trainable is True, the variable will be trainable=False. Otherwise,
        the default value of the network is used"""
        return tf.compat.v1.get_variable(name, shape, trainable=self.trainable and not not_trainable)

    def get_layer_name(self):
        return layer_name
    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer 
    def identity(self, input, name):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                list_output.append(tf.identity(input[0][iter_gpu], name=name))
        return list_output

    @layer
    def zero_padding(self, input, paddings, name):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
                    list_output.append(tf.pad(input[0][iter_gpu], paddings=pad_mat, name=name))
        return list_output

    @layer
    def conv(self,
             input,
             kernel_h,
             kernel_w,
             num_channels_out,
             stride_h,
             stride_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             not_trainable=False):
        """
        Strided CONV layer

        :param input: Skip this argument, as it is provided automatically
        :param kernel_h: Spatial height of kernel
        :param kernel_w: Spatial width of kernel
        :param num_channels_out:
        :param stride_h: Spatial stride in vertical direction
        :param stride_w: Spatial stride in horizontal direction
        :param name:
        :param relu: Whether ReLU activation should be appended after CONV and (optional) biasing
        :param padding: Padding type 'SAME' or 'VALID'
        :param group:
        :param biased: Whether channel-wise bias should be applied (same bias for all pixels in one channel) after CONV
        :param not_trainable: Wheather variables needed for this layer should never be trainable
        :return: Layer output
        """
        # Verify that the padding is acceptable
        self.validate_padding(padding)

        # Get the number of channels in the input
        num_channels_in = input[0][0].get_shape()[-1]
        # Scale usual amount of channels to reduce computation
        if 'out' not in name and 'cls' not in name:
            num_channels_out *= self.filter_scale

        list_output = [[]]*len(input[0])  
        for iter_gpu in np.random.permutation(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    kernel = self.make_var('weights', [kernel_h, kernel_w, num_channels_in, num_channels_out], not_trainable)
                    output = tf.nn.conv2d(input[0][iter_gpu], kernel, [1, stride_h, stride_w, 1], padding=padding,
                                          data_format=DEFAULT_DATAFORMAT)

                    if biased:
                        biases = self.make_var('biases', [num_channels_out], not_trainable)
                        output = tf.nn.bias_add(output, biases)
                    if relu:
                        output = tf.nn.relu(output, name=scope.name)

                    list_output[iter_gpu] = output

        return list_output

    @layer
    def atrous_conv(self,
                    input,
                    kernel_h,
                    kernel_w,
                    num_channels_out,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True,
                    not_trainable=False):
        """
        Atrous CONV layer, also called dilated CONV

        :param input: Skip this argument, as it is provided automatically
        :param kernel_h: Spatial height of kernel
        :param kernel_w: Spatial width of kernel
        :param num_channels_out:
        :param dilation: Dilation factor
        :param name:
        :param relu: Whether ReLU activation should be appended after CONV and (optional) biasing
        :param padding: Padding type 'SAME' or 'VALID'
        :param group:
        :param biased: Whether channel-wise bias should be applied (same bias for all pixels in one channel) after CONV
        :param not_trainable: Wheather variables needed for this layer should never be trainable
        :return: Layer output
        """

        list_output = [[]]*len(input[0])  
        for iter_gpu in np.random.permutation(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    # Verify that the padding is acceptable
                    self.validate_padding(padding)

                    # Get the number of channels in the input
                    num_channels_in = input[0][iter_gpu].get_shape()[-1]
                    # Scale usual amount of channels to reduce computation
                    num_channels_out *= self.filter_scale

                    kernel = self.make_var('weights', [kernel_h, kernel_w, num_channels_in, num_channels_out], not_trainable)
                    output = tf.nn.atrous_conv2d(input[0][iter_gpu], kernel, dilation, padding=padding)

                    if biased:
                        biases = self.make_var('biases', [num_channels_out], not_trainable)
                        output = tf.nn.bias_add(output, biases)
                    if relu:
                        output = tf.nn.relu(output, name=scope.name)
                    
                    list_output[iter_gpu] = output

        return list_output


    @layer
    def convlstm(self, input, timeSequence, kernel_h, kernel_w, num_channels_out, name, convtype='convolution'):
        """
        convLSTM Unit. The input must include the passed amount of time steps within the batch and the recurrent unit
        will be unrolled for this amount of time steps (i.e. this is how far into the past BPTT will be executed)
        """
        # Get input dimensions
        num_channels_in = input[0][0].get_shape()[-1]
        input_h = input[0][0].get_shape()[1]
        input_w = input[0][0].get_shape()[2]
        batch_size = input[0][0].get_shape()[0]

        # Scale usual amount of channels to reduce computation
        num_channels_out *= self.filter_scale

        print ('time_sequence length: {}'.format(timeSequence))

        list_output = [[]]*len(input[0])  
        for iter_gpu in np.random.permutation(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    input_reshape = tf.reshape(input[0][iter_gpu], [-1, timeSequence, input_h, input_w, num_channels_in])
                    #lstm = rnn.ConvLSTMCell(conv_ndims=2, input_shape=[input_h, input_w, num_channels_in],
                    #                        output_channels=num_channels_out, kernel_shape=[kernel_h, kernel_w], name=name)
                    lstm = ConvLSTMCellAP(conv_ndims=2, input_shape=[input_h, input_w, num_channels_in],
                                          num_channels_out=num_channels_out, kernel_shape=[kernel_h, kernel_w], name=name,
                                          convtype=convtype)

                    USE_TRAINABLE_INITIAL_CELL_STATE = False
                    if USE_TRAINABLE_INITIAL_CELL_STATE:
                        # Create initial state as (trainable) variable
                        initial_state = self.make_variable_initial_lstm_state(batch_size//timeSequence, lstm.state_size)
                        print([state for state in initial_state])

                        output, states = tf.nn.dynamic_rnn(lstm, input_reshape, initial_state=initial_state, dtype=tf.float32)
                    else:
                        output, states = tf.nn.dynamic_rnn(lstm, input_reshape, dtype=tf.float32)

                    list_output[iter_gpu] = tf.reshape(output, [batch_size, input_h, input_w, -1])

        return list_output

    @layer
    def relu(self, input, name):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.nn.relu(input[0][iter_gpu], name=name))
        return list_output

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.nn.max_pool2d(input[0][iter_gpu],
                                          ksize=[1, k_h, k_w, 1],
                                          strides=[1, s_h, s_w, 1],
                                          padding=padding,
                                          name=name,
                                          data_format=DEFAULT_DATAFORMAT))
        return list_output

    @layer
    def max_pool_with_argmax(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                list_output.append(tf.nn.max_pool_with_argmax(input[0][iter_gpu],
                                      ksize=[1, k_h, k_w, 1],
                                      strides=[1, s_h, s_w, 1],
                                      padding=padding,
                                      name=name))
        return list_output

    def split_max_pool_with_argmax(self):
        buff = self.get_output()
        maxpool_ind = []
        maxpool_pos = []
        for iter_gpu in range(len(buff)):
            maxpool_ind.append(buff[iter_gpu][0])
            maxpool_pos.append(buff[iter_gpu][1])
        return maxpool_ind, maxpool_pos

    @layer 
    def unpool_with_argmax(self, inputs, k_h, k_w, output_shape, name):
        input_shape = inputs[0][0].get_shape().as_list()
        ksize=[1, k_h, k_w, 1]
        #output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    pool_ = tf.reshape(inputs[0][iter_gpu], [flat_input_size])
                    batch_range = tf.reshape(tf.range(output_shape[0], dtype=inputs[1][iter_gpu].dtype), shape=[input_shape[0], 1, 1, 1])
                    b = tf.ones_like(inputs[1][iter_gpu]) * batch_range
                    b = tf.reshape(b, [flat_input_size, 1])
                    ind_ = tf.reshape(inputs[1][iter_gpu], [flat_input_size, 1])
                    ind_ = tf.concat([b, ind_], 1)

                    output = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
                    output = tf.reshape(output, output_shape)
                    list_output.append(output)
        return list_output

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    self.validate_padding(padding)
                    list_output.append(tf.nn.avg_pool2d(input[0][iter_gpu],
                                          ksize=[1, k_h, k_w, 1],
                                          strides=[1, s_h, s_w, 1],
                                          padding=padding,
                                          name=name,
                                          data_format=DEFAULT_DATAFORMAT))
        return list_output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.nn.local_response_normalization(input[0][iter_gpu],
                                                                          depth_radius=radius,
                                                                          alpha=alpha,
                                                                          beta=beta,
                                                                          bias=bias,
                                                                          name=name))
        return list_output

    @layer
    def concat(self, inputs, axis, name):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    inputs_one = []
                    for ix in range(len(inputs)):
                        inputs_one.append(inputs[ix][iter_gpu])
                    list_output.append(tf.concat(axis=axis, values=inputs_one, name=name))
        return list_output


    @layer
    def stack(self, inputs, axis, name):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    inputs_one = []
                    for ix in range(len(inputs)):
                        inputs_one.append(inputs[ix][iter_gpu])
                    list_output.append(tf.stack(values=inputs_one, axis=axis, name=name))
        return list_output

    @layer
    def add(self, inputs, name):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    inputs_one = []
                    for ix in range(len(inputs)):
                        inputs_one.append(inputs[ix][iter_gpu])
                    inputs_one[0] = tf.compat.v1.image.resize_bilinear(inputs_one[0], size=tf.shape(inputs_one[1])[1:3])
        
                    list_output.append(tf.add_n(inputs_one, name=name))
        return list_output

    @layer
    def multiply_2tensors(self, inputs, name):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.multiply(inputs[0][iter_gpu], inputs[1][iter_gpu]))
        return list_output

    @layer
    def fc(self, input, num_out, name, relu=True, not_trainable=False):
        """Fully-connected layer with optional ReLU activation"""
        list_output = [[]]*len(input[0])  
        for iter_gpu in np.random.permutation(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    input_shape = input[0].get_shape()
                    if input_shape.ndims == 4:
                        # The input is spatial. Vectorize it first.
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input[0][iter_gpu], [-1, dim])  # vectorizes each batch of input
                    else:
                        feed_in = input[0][iter_gpu]
                        dim = input_shape[-1].value

                    weights = self.make_var('weights', [dim, num_out], not_trainable)
                    biases = self.make_var('biases', [num_out], not_trainable)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    list_output[iter_gpu] = op(feed_in, weights, biases, name=scope.name)
        return list_output


    @layer 
    def batch_normalization(self, input, name, not_trainable=False, use_gamma=True, use_beta=True, scale_offset=True, relu=False, momentum=0.95, epsilon=1e-5):
        float_type = tf.float32
        def get_bn_variables(n_out, use_scale, use_bias):
            if use_bias:
                beta = tf.compat.v1.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0), trainable=True, dtype=float_type)
            else:
                beta = tf.zeros([n_out], name='beta')
            if use_scale:
                gamma = tf.compat.v1.get_variable('gamma', [n_out], initializer=tf.constant_initializer(1.0), trainable=True, dtype=float_type)
            else:
                gamma = tf.ones([n_out], name='gamma')
            # x * gamma + beta

            moving_mean = tf.compat.v1.get_variable('moving_mean', [n_out], initializer=tf.constant_initializer(), trainable=False, dtype=float_type)
            moving_var = tf.compat.v1.get_variable('moving_variance', [n_out], initializer=tf.constant_initializer(1), trainable=False, dtype=float_type)
            return beta, gamma, moving_mean, moving_var

        def update_bn_ema(output, batch_mean, batch_var, moving_mean, moving_var, decay):
            from tensorflow.contrib.framework import add_model_variable
            from tensorflow.python.training import moving_averages

            update_op1 = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=False, name='mean_ema_op')
            update_op2 = moving_averages.assign_moving_average(moving_var, batch_var, decay, zero_debias=False, name='var_ema_op')
            add_model_variable(moving_mean)
            add_model_variable(moving_var)

            # seems faster than delayed update, but might behave otherwise in distributed settings.
            tf.compat.v1.add_to_collections(tf.compat.v1.GraphKeys.UPDATE_OPS, update_op1)
            tf.compat.v1.add_to_collections(tf.compat.v1.GraphKeys.UPDATE_OPS, update_op2)
            return tf.identity(output)


        # get shape

        shape = input[0][0].get_shape().as_list()
        assert len(shape) in [2, 4]
        n_out = shape[-1]
        
        # Batch normalization for multiple gpus

        means = []
        square_means = []
        apply_onGPU = np.random.choice(len(input[0]))
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                means.append(tf.reduce_mean(input[0][iter_gpu], [0, 1, 2]))
                square_means.append(tf.reduce_mean(tf.square(input[0][iter_gpu]), [0, 1, 2]))

        with tf.device(assign_to_device('/gpu:%d' % apply_onGPU, controller)):
        #with tf.device('/cpu:%d' % 0):
            shape = tf.shape(input[0][0])
            num = shape[0] * shape[1] * shape[2] * len(input[0])
            batch_mean = tf.reduce_mean(means, axis=0)
            batch_var = tf.reduce_mean(square_means, axis=0) - tf.square(batch_mean)
            batch_var *= tf.cast(num, float_type) / tf.cast(num-1, float_type)  # unbiased variance
            

        list_output = [[]]*len(input[0])  
        for iter_gpu in np.random.permutation(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
                    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta)
                    
                    if self.is_inference:
                        batch_mean = moving_mean
                        batch_var = moving_var
                    else:
                        batch_mean = tf.cond(self.is_training,
                                            lambda: batch_mean,
                                            lambda: moving_mean)
                        batch_var = tf.cond(self.is_training,
                                            lambda: batch_var,
                                            lambda: moving_var)

                    output = tf.nn.batch_normalization(input[0][iter_gpu], batch_mean, batch_var, beta, gamma, epsilon)

                    if relu:
                        output = tf.nn.relu(output)

                    if self.is_inference:
                        list_output[iter_gpu] = output
                    else:
                        if iter_gpu == apply_onGPU:
                            # gathering stats in the main gpu device only.
                            output = tf.cond(self.is_training,
                                            lambda: update_bn_ema(output, batch_mean, batch_var, moving_mean, moving_var, momentum),
                                            lambda: tf.identity(output))
                            list_output[iter_gpu] = output
                            #list_output[iter_gpu] = update_bn_ema(output, batch_mean, batch_var, moving_mean, moving_var, momentum)
                        else: 
                            list_output[iter_gpu] = output   

        return list_output

    @layer
    def layer_normalization(self, input, name, not_trainable=False, relu=False):
        """
        Layer Normalization normalizes feature maps, i.e. mean and variance are calculated over all spatial positions
        and all channels per sample in batch (for comparison: Batch Normalization calculates mean and variance over all
        spatial positions and all samples in batch per channel). Afterwards each channel is scaled and shifted according
        to two learnable values (beta and gamma) per channel (same as in BatchNorm).

        If requested, a ReLU is appended at the end.
        """
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    output = layers.layer_norm(input[0][iter_gpu], trainable=self.trainable and not not_trainable, scope=name)

                    if relu:
                        output = tf.nn.relu(output)
                    list_output.append(output)

        return list_output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.nn.dropout(input, keep, name=name))
        return list_output

    @layer
    def resize_bilinear(self, input, size, name):
        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.compat.v1.image.resize_bilinear(input[0][iter_gpu], size=size, align_corners=True, name=name))
        return list_output

    @layer
    def reduce_mean(self, input, axis, name, keep_dims=True):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.reduce_mean(input[0][iter_gpu], axis, keep_dims=keep_dims))
        return list_output()

    @layer
    def sigmoid(self, input, name):
        list_output = []
        for iter_gpu in range(len(inputs[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    list_output.append(tf.sigmoid(input[0][iter_gpu]))
        return list_output

    @layer
    def interp(self, input, s_factor=1, z_factor=1, name=None):

        list_output = []
        for iter_gpu in range(len(input[0])):
            with tf.device(assign_to_device('/gpu:%d' % iter_gpu, controller)):
                #with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
                    ori_h, ori_w = input[0][iter_gpu].get_shape().as_list()[1:3]
                    # shrink
                    ori_h = (ori_h - 1) * s_factor + 1
                    ori_w = (ori_w - 1) * s_factor + 1
                    # zoom
                    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
                    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)
                    resize_shape = [int(ori_h), int(ori_w)]

                    list_output.append(tf.compat.v1.image.resize_bilinear(input[0][iter_gpu], size=resize_shape, align_corners=True, name=name))

        return list_output



