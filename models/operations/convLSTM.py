"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf

# TODO only tested for 2D convolution!!!


class ConvLSTMCellAP(rnn_cell_impl.RNNCell):
    """
    Convolutional LSTM recurrent network cell. Implementation based on class tf.contrib.rnn.ConvLSTMCell.
    """

    def __init__(self,
                 conv_ndims,
                 input_shape,
                 num_channels_out,
                 kernel_shape,
                 convtype='convolution',
                 channel_multiplier=1,
                 use_bias=True,
                 forget_bias=1.0,
                 initializers=None,
                 name="conv_lstm_cell"):
        """
        Construct ConvLSTMCellAP.

        Args:
            conv_ndims: Convolution dimensionality (1, 2 or 3)
            input_shape: Shape of the input as int tuple, excluding batch size and time step. E.g. (height, width,
                         num_channels) for images
            num_channels_out: Number of output channels of the convLSTM
            kernel_shape: Shape of the kernels as int tuple (of size 1, 2 or 3).
            convtype: convLSTM type - 'convolution': standard convLSTM layer
                                    - 'spatial': convolution is separated spatial (n,n) = (n,1) + (1,n)
                                    - 'depthwise': convolution is separated depthwise
                                    - 'separable': depthwise separable convolution (after depth-wise CONV,
                                    a 1x1 convolution is applied over all channels)
            channel_multiplier: Channel multiplier for depthwise CONVs
            use_bias: Whether to use bias in convolutions
            initializers: Unused
            name: Name of the module
        Raises:
            ValueError: If `input_shape` is incompatible with `conv_ndims` or chose type of convolution
        """
        super(ConvLSTMCellAP, self).__init__(name=name)

        if conv_ndims != len(input_shape) - 1:
            raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(input_shape, conv_ndims))

        self._conv_ndims = conv_ndims
        self._input_shape = input_shape
        self._num_channels_out = num_channels_out
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._convtype = convtype
        self._channel_multiplier = channel_multiplier
        self._forget_bias = forget_bias

        self._total_output_channels = num_channels_out

        self._output_size = tensor_shape.TensorShape(self._input_shape[:-1] + [self._total_output_channels])
        cell_state_size = tensor_shape.TensorShape(self._input_shape[:-1] + [self._num_channels_out])
        self._state_size = rnn_cell_impl.LSTMStateTuple(cell_state_size, self._output_size)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state, scope=None):
        cell, hidden = state  # split state tupel in last cell state c_t-1 and last hidden state h_t-1

        use_unoptimized_convs = True

        new_hidden = _convs_unoptimized([inputs, hidden], self._kernel_shape, 4 * self._num_channels_out,
                                        self._use_bias, convtype=self._convtype)
        # Channels of new_hidden are concatenation of tensors for different gates and intermediate result for next
        # cell state -> split into those tensors
        gates = array_ops.split(value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)
        input_gate, new_input, forget_gate, output_gate = gates

        new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
        new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)

        output = math_ops.sigmoid(output_gate) * math_ops.tanh(new_cell)

        new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)

        return output, new_state



class Conv1DLSTMCell(ConvLSTMCellAP):
    """
    1D Convolutional LSTM recurrent network cell.
    https://arxiv.org/pdf/1506.04214v1.pdf
    """

    def __init__(self, name="conv_1d_lstm_cell", **kwargs):
        """Construct Conv1DLSTM. See `ConvLSTMCellAP` for more details."""
        super(Conv1DLSTMCell, self).__init__(conv_ndims=1, name=name, **kwargs)


class Conv2DLSTMCell(ConvLSTMCellAP):
    """
    2D Convolutional LSTM recurrent network cell.
    https://arxiv.org/pdf/1506.04214v1.pdf
    """

    def __init__(self, name="conv_2d_lstm_cell", **kwargs):
        """Construct Conv2DLSTM. See `ConvLSTMCellAP` for more details."""
        super(Conv2DLSTMCell, self).__init__(conv_ndims=2, name=name, **kwargs)


class Conv3DLSTMCell(ConvLSTMCellAP):
    """
    3D Convolutional LSTM recurrent network cell.
    https://arxiv.org/pdf/1506.04214v1.pdf
    """

    def __init__(self, name="conv_3d_lstm_cell", **kwargs):
        """Construct Conv3DLSTM. See `ConvLSTMCellAP` for more details."""
        super(Conv3DLSTMCell, self).__init__(conv_ndims=3, name=name, **kwargs)


def _convs_unoptimized(args, filter_size, num_features, bias, bias_start=0.0, convtype='convolution'):
    """
    Convolution of concatenated tensors args with kernel of filter_size and num_features output channels
    (ACTUALLY THIS IS NOT TRUE FOR ANY TYPE EXCEPT STANDARD CONV!)

    Args:
        args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D, batch x n, Tensors.
        filter_size: int tuple of filter height and width.
        num_features: int, number of features.
        bias: Whether to use biases in the convolution layer.
        bias_start: starting value to initialize the bias; 0 by default.
        convtype: convLSTM type - 'convolution': standard convLSTM layer
                                - 'spatial': convolution is separated spatial (n,n) = (n,1) + (1,n)
                                - 'depthwise': convolution is separated depthwise
                                - 'separable': depthwise separable convolution (after depth-wise conv, a 1x1 convolution
                                 is applied over all channels)
    Returns:
        A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1

    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    shape_length = len(shapes[0])
    for shape in shapes:
        if len(shape) not in [3, 4, 5]:
            raise ValueError("Conv Linear expects 3D, 4D or 5D arguments: %s" % str(shapes))
        if len(shape) != len(shapes[0]):
            raise ValueError("Conv Linear expects all args to be of same Dimension: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[-1]
    dtype = [a.dtype for a in args][0]

    if shape_length != 4 and convtype == "separable":
        print ('[ERROR] separable convLSTM is only implemented for conv2D')
        raise NotImplementedError 

    if len(args) != 2:
            print ('LSTM is only implemented with len(args) = 2!')
            raise NotImplementedError

    # Determine correct conv operation

    c_i = shapes[0][-1]  # number of input channels per tensor in args
    c_o = num_features//4  # number of output channels per gate and cell state

    if convtype == 'separable': 
        if shape_length == 3:
            conv_op = tf.nn.separable_conv1d  # ? does not exist
            strides = 1
        elif shape_length == 4:
            conv_op = tf.nn.separable_conv2d
            strides = shape_length * [1]
        elif shape_length == 5:
            conv_op = tf.nn.separable_conv3d  # ? does not exist
            strides = shape_length * [1]
        else:
            raise NotImplementedError
        channel_multiplier = 1
    elif convtype == 'depthwise': 
        if shape_length == 3:
            conv_op = tf.nn.depthwise_conv1d  # ? does not exist
            strides = 1
        elif shape_length == 4:
            conv_op = tf.nn.depthwise_conv2d
            strides = shape_length * [1]
        elif shape_length == 5:
            conv_op = tf.nn.depthwise_conv3d  # ? does not exist
            strides = shape_length * [1]
        else:
            raise NotImplementedError
        channel_multiplier = 1
    else:  # Normal CONV and spatially separable CONV
        if shape_length == 3:
            conv_op = nn_ops.conv1d
            strides = 1
        elif shape_length == 4:
            conv_op = nn_ops.conv2d
            strides = shape_length * [1]
        elif shape_length == 5:
            conv_op = nn_ops.conv3d
            strides = shape_length * [1]
        else:
            raise NotImplementedError

    # Now the computation

    if convtype == 'spatial':
        # Get kernels

        kernel_h = vs.get_variable("kernel_h", [filter_size[0], 1, total_arg_size_depth, num_features], dtype=dtype)
        print('kernel_h: ', [filter_size[0], 1, total_arg_size_depth, num_features])
        kernel_w = vs.get_variable("kernel_w", [1, filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        print('kernel_w: ', [1, filter_size[1], total_arg_size_depth, num_features])

        W_ix_h = kernel_h[..., 0:c_i, 0:1*c_o]  # Name pattern: W(eights) for i(nput gate) for h(eight) CONV with x
        W_ih_h = kernel_h[..., c_i:2*c_i, 0:1*c_o]
        W_cx_h = kernel_h[..., 0:c_i, 1*c_o:2*c_o]
        W_ch_h = kernel_h[..., c_i:2*c_i, 1*c_o:2*c_o]
        W_fx_h = kernel_h[..., 0:c_i, 2*c_o:3*c_o]
        W_fh_h = kernel_h[..., c_i:2*c_i, 2*c_o:3*c_o]
        W_ox_h = kernel_h[..., 0:c_i, 3*c_o:4*c_o]
        W_oh_h = kernel_h[..., c_i:2*c_i, 3*c_o:4*c_o]

        W_ix_w = kernel_w[..., 0:c_i, 0:1*c_o]
        W_ih_w = kernel_w[..., c_i:2*c_i, 0:1*c_o]
        W_cx_w = kernel_w[..., 0:c_i, 1*c_o:2*c_o]
        W_ch_w = kernel_w[..., c_i:2*c_i, 1*c_o:2*c_o]
        W_fx_w = kernel_w[..., 0:c_i, 2*c_o:3*c_o]
        W_fh_w = kernel_w[..., c_i:2*c_i, 2*c_o:3*c_o]
        W_ox_w = kernel_w[..., 0:c_i, 3*c_o:4*c_o]
        W_oh_w = kernel_w[..., c_i:2*c_i, 3*c_o:4*c_o]

        # input gate

        i_x_h = conv_op(args[0], W_ix_h, strides, padding="SAME")
        i_x = conv_op(i_x_h, W_ix_w, strides, padding="SAME")
        i_h_h = conv_op(args[1], W_ih_h, strides, padding="SAME")
        i_h = conv_op(i_h_h, W_ih_w, strides, padding="SAME")

        # new input (= intermediate step for new cell state)

        c_x_h = conv_op(args[0], W_cx_h, strides, padding="SAME")
        c_x = conv_op(c_x_h, W_cx_w, strides, padding="SAME")
        c_h_h = conv_op(args[1], W_ch_h, strides, padding="SAME")
        c_h = conv_op(c_h_h, W_ch_w, strides, padding="SAME")

        # forget gate

        f_x_h = conv_op(args[0], W_fx_h, strides, padding="SAME")
        f_x = conv_op(f_x_h, W_fx_w, strides, padding="SAME")
        f_h_h = conv_op(args[1], W_fh_h, strides, padding="SAME")
        f_h = conv_op(f_h_h, W_fh_w, strides, padding="SAME")

        # output gate

        o_x_h = conv_op(args[0], W_ox_h, strides, padding="SAME")
        o_x = conv_op(o_x_h, W_ox_w, strides, padding="SAME")
        o_h_h = conv_op(args[1], W_oh_h, strides, padding="SAME")
        o_h = conv_op(o_h_h, W_oh_w, strides, padding="SAME")

        # sum up results
        
        res_x = array_ops.concat(axis=shape_length - 1, values=[i_x, c_x, f_x, o_x])
        res_h = array_ops.concat(axis=shape_length - 1, values=[i_h, c_h, f_h, o_h])
        res = tf.add(res_x, res_h)

    elif convtype == 'depthwise':
        # Get kernels

        kernel_depth = vs.get_variable("kernel_depth", filter_size + [total_arg_size_depth, 4*channel_multiplier],
                                       dtype=dtype)
        print('kernel_depth: ', filter_size + [total_arg_size_depth, 4*channel_multiplier])

        W_ix = kernel_depth[..., 0:c_i, 0:1*channel_multiplier]
        W_ih = kernel_depth[..., c_i:2*c_i, 0:1*channel_multiplier]
        W_cx = kernel_depth[..., 0:c_i, 1*channel_multiplier:2*channel_multiplier]
        W_ch = kernel_depth[..., c_i:2*c_i, 1*channel_multiplier:2*channel_multiplier]
        W_fx = kernel_depth[..., 0:c_i, 2*channel_multiplier:3*channel_multiplier]
        W_fh = kernel_depth[..., c_i:2*c_i, 2*channel_multiplier:3*channel_multiplier]
        W_ox = kernel_depth[..., 0:c_i, 3*channel_multiplier:4*channel_multiplier]
        W_oh = kernel_depth[..., c_i:2*c_i, 3*channel_multiplier:4*channel_multiplier]

        # input gate

        i_x = conv_op(args[0], W_ix, strides, padding="SAME")
        i_h = conv_op(args[1], W_ih, strides, padding="SAME")

        # new input (= intermediate step for new cell state)

        c_x = conv_op(args[0], W_cx, strides, padding="SAME")
        c_h = conv_op(args[1], W_ch, strides, padding="SAME")

        # forget gate

        f_x = conv_op(args[0], W_fx, strides, padding="SAME")
        f_h = conv_op(args[1], W_fh, strides, padding="SAME")

        # output gate

        o_x = conv_op(args[0], W_ox, strides, padding="SAME")
        o_h = conv_op(args[1], W_oh, strides, padding="SAME")

        # sum up results
        
        res_x = array_ops.concat(axis=shape_length - 1, values=[i_x, c_x, f_x, o_x])
        res_h = array_ops.concat(axis=shape_length - 1, values=[i_h, c_h, f_h, o_h])
        res = tf.add(res_x, res_h)

    elif convtype == 'separable':
        # Get kernels

        kernel_depth = vs.get_variable("kernel_depth", filter_size + [total_arg_size_depth, 4*channel_multiplier],
                                       dtype=dtype)
        print('kernel_depth: ', filter_size + [total_arg_size_depth, 4*channel_multiplier])
        kernel_sep = vs.get_variable("kernel_sep", [1, 1, total_arg_size_depth, num_features], dtype=dtype)
        print('kernel_sep: ', [1, 1, total_arg_size_depth, num_features])

        W_ix = kernel_depth[..., 0:c_i, 0:1*channel_multiplier]
        W_ih = kernel_depth[..., c_i:2*c_i, 0:1*channel_multiplier]
        W_cx = kernel_depth[..., 0:c_i, 1*channel_multiplier:2*channel_multiplier]
        W_ch = kernel_depth[..., c_i:2*c_i, 1*channel_multiplier:2*channel_multiplier]
        W_fx = kernel_depth[..., 0:c_i, 2*channel_multiplier:3*channel_multiplier]
        W_fh = kernel_depth[..., c_i:2*c_i, 2*channel_multiplier:3*channel_multiplier]
        W_ox = kernel_depth[..., 0:c_i, 3*channel_multiplier:4*channel_multiplier]
        W_oh = kernel_depth[..., c_i:2*c_i, 3*channel_multiplier:4*channel_multiplier]

        Wsep_ix = kernel_sep[..., 0:c_i, 0:1*c_o]
        Wsep_ih = kernel_sep[..., c_i:2*c_i, 0:1*c_o]
        Wsep_cx = kernel_sep[..., 0:c_i, 1*c_o:2*c_o]
        Wsep_ch = kernel_sep[..., c_i:2*c_i, 1*c_o:2*c_o]
        Wsep_fx = kernel_sep[..., 0:c_i, 2*c_o:3*c_o]
        Wsep_fh = kernel_sep[..., c_i:2*c_i, 2*c_o:3*c_o]
        Wsep_ox = kernel_sep[..., 0:c_i, 3*c_o:4*c_o]
        Wsep_oh = kernel_sep[..., c_i:2*c_i, 3*c_o:4*c_o]

        # input gate

        i_x = conv_op(args[0], W_ix, Wsep_ix, strides, padding="SAME")
        i_h = conv_op(args[1], W_ih, Wsep_ih, strides, padding="SAME")

        # new input (= intermediate step for new cell state)

        c_x = conv_op(args[0], W_cx, Wsep_cx, strides, padding="SAME")
        c_h = conv_op(args[1], W_ch, Wsep_ch, strides, padding="SAME")

        # forget gate

        f_x = conv_op(args[0], W_fx, Wsep_fx, strides, padding="SAME")
        f_h = conv_op(args[1], W_fh, Wsep_fh, strides, padding="SAME")

        # output gate

        o_x = conv_op(args[0], W_ox, Wsep_ox, strides, padding="SAME")
        o_h = conv_op(args[1], W_oh, Wsep_oh, strides, padding="SAME")

        # sum up results
        
        res_x = array_ops.concat(axis=shape_length - 1, values=[i_x, c_x, f_x, o_x])
        res_h = array_ops.concat(axis=shape_length - 1, values=[i_h, c_h, f_h, o_h])
        res = tf.add(res_x, res_h)

    else:  # normal CONV
        # Get kernel

        kernel = vs.get_variable("kernel", filter_size + [total_arg_size_depth, 4*c_o], dtype=dtype)
        print('kernel: ', filter_size + [total_arg_size_depth, 4*c_o])

        W_ix = kernel[..., 0:c_i, 0:1*c_o]
        W_ih = kernel[..., c_i:2*c_i, 0:1*c_o]
        W_cx = kernel[..., 0:c_i, 1*c_o:2*c_o]
        W_ch = kernel[..., c_i:2*c_i, 1*c_o:2*c_o]
        W_fx = kernel[..., 0:c_i, 2*c_o:3*c_o]
        W_fh = kernel[..., c_i:2*c_i, 2*c_o:3*c_o]
        W_ox = kernel[..., 0:c_i, 3*c_o:4*c_o]
        W_oh = kernel[..., c_i:2*c_i, 3*c_o:4*c_o]

        # input gate

        i_x = conv_op(args[0], W_ix, strides, padding="SAME")
        i_h = conv_op(args[1], W_ih, strides, padding="SAME")

        # new input (= intermediate step for new cell state)

        c_x = conv_op(args[0], W_cx, strides, padding="SAME")
        c_h = conv_op(args[1], W_ch, strides, padding="SAME")

        # forget gate

        f_x = conv_op(args[0], W_fx, strides, padding="SAME")
        f_h = conv_op(args[1], W_fh, strides, padding="SAME")

        # output gate

        o_x = conv_op(args[0], W_ox, strides, padding="SAME")
        o_h = conv_op(args[1], W_oh, strides, padding="SAME")

        # sum up results
        
        res_x = array_ops.concat(axis=shape_length - 1, values=[i_x, c_x, f_x, o_x])
        res_h = array_ops.concat(axis=shape_length - 1, values=[i_h, c_h, f_h, o_h])
        res = tf.add(res_x, res_h)
  
    if not bias:
        return res
    bias_term = vs.get_variable("biases", [num_features], dtype=dtype,
                                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term

