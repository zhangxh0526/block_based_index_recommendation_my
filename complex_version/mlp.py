"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""

import copy
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tf_ops import glorot, ones, zeros

class MLP(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 act_fn, scope='mlp'):

        self.inputs = inputs

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim

        self.act_fn = act_fn
        self.scope = scope


        # initialize message passing transformation parameters
        # h: x -> x'
        self.weights, self.bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)

        # graph message passing
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.inputs

        # raise x into higher dimension
        for l in range(len(self.weights)):
            x = tf.matmul(x, self.weights[l])
            x += self.bias[l]
            x = self.act_fn(x)


        return x
