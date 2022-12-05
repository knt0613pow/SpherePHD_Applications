import torch
import numpy as np
from torch.nn import init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import Module
import math

class PHD_conv2d(Module):
    def __init__(self, 
               in_dim:int,
               out_dim:int,
               stride : int,
               device = None,
               dtype = None,
               activation = 'ReLU',
               ) -> None:
        super(PHD_conv2d, self).__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.activation = getattr(torch.nn, activation)()
        shape = np.asarray([out_dim, in_dim, 10, 1])
        self.weight = Parameter(torch.empty(*shape),)# **factory_kwargs)
        self.bias = Parameter(torch.empty(shape[0]),) #**factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, conv_table):
        x = input[..., conv_table].squeeze()
        output = F.conv2d(x,self.weight, self.bias, self.stride)
        output = self.activation(output)
        return output
    
def PHD_maxpool(input, adj_table, pooling_table):
    '''
    input (N, C, H =1, W = 20*(4 ** division))
    adj_table (W, 4)
    pooling_table (20*(4**(division-1)))
'''
    
    output = input[..., adj_table].squeeze() # (N, C, 4, W,)
    output = output[..., pooling_table] # (N, C, 4, W/4, )
    output = torch.max(output, dim=2,keepdim=True)[0] # (N, C/ 1, W/4)
    return output

def PHD_avgpool(input):
    '''
    input (N, C, H =1, W = 20*(4 ** division))
    '''
    output = torch.mean(input, (2, 3)) # (N, C)
    return output    


'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def conv2d(input, in_dim, out_dim, conv_table, name, 
           reuse=False, stride=1, activation='elu', padding='VALID'):

    shape = np.asarray([10, 1, in_dim, out_dim])
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        # width = 20 * 4**subdivision
        # input = (n_batch, 1, width, n_channel), conv_table (kernelsize, width)
        # Turns x into an array of shape (n_batch, 1, kernel=10, width, n_channel)
        x = tf.gather(input, conv_table, axis=2)
        # squeeze the array into shape (n_batch, kernel=10, width, n_channel)
        x = tf.squeeze(x, axis=1)
        output = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding) + bias
        if activation == 'elu':
            output = tf.nn.elu(output)
        
        return output

def channel_conv(input, in_dim, out_dim, name, reuse=False, 
                 stride=1, activation='elu', padding='VALID'):

    shape = np.asarray([1, 1, in_dim, out_dim])
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        # input = (n_batch, 1, width, n_channel)
        output = tf.nn.conv2d(input, weight, [1, stride, stride, 1], padding) + bias

        if activation == 'elu':
            output = tf.nn.elu(output)
        return output

def maxpool(x, adj_table, pooling_table):

    # input = (n_batch, 1, width, n_channel)
    # Turns x into an array of shape (n_batch, 1, kernel=4, width, n_channel)
    x = tf.gather(x, adj_table, axis=2)
    # squeeze the array into shape (n_batch, kernel=4, width, n_channel)
    x = tf.squeeze(x, axis=1)
    # pool_table make the subdivision-1, with size: (width/4)
    # Picks out correct pool indexes (n_batch, kernel=4, width/4, n_channel)
    x = tf.gather(x, pooling_table, axis=2)
    # Max from pool (n_batch, 1, width/4, n_channel)
    x = tf.reduce_max(x, axis=1, keepdims=True)
    return x

def upsample(x, upsample_table):
    # input = (n_batch, 1, width, n_channel)
    # Turns x into an array of shape (n_batch, 1, width*4, n_channel)
    x = tf.keras.layers.UpSampling2D((1, 4))(x)
    # unpooling shape (n_batch, 1, width*4, n_channel)
    x = tf.gather(x, upsample_table, axis=2)
    return x

def avgpool(x):
    x = tf.reduce_mean(x, axis=[1, 2])
    return x
'''

      