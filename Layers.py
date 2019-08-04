import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, X):
        pass


class ConvLayer(Layer):
    def __init__(self, id, M1, M2, activation):
        self.activation = activation
        self.id = id
        filter_shape = [3, 3, M1, M2]

        W_init = self.__init_weight(filter_shape)
        b_init = tf.zeros(M2, dtype=tf.float32)

        self.W = tf.Variable(W_init, name="W" + id)
        self.b = tf.Variable(b_init, name="b" + id)

    
    def forward(self, X, training=True):
        conv_out = tf.nn.conv2d(X, self.W, [1,1,1,1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        return self.activation(conv_out)


    def __init_weight(self, shape):
        return tf.random.normal(shape, dtype=tf.float32)


class PoolingLayer(Layer):
    def forward(self, X, training=True):
        return tf.nn.max_pool(X, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


class VanillaLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.activation = activation
        self.id = id

        W_init = tf.random.normal((M1, M2), dtype=tf.float32)
        b_init = tf.zeros((M2,), dtype=tf.float32)
        
        self.W = tf.Variable(W_init, name="W" + id)
        self.b = tf.Variable(b_init, name="b" + id)
        
    def forward(self, X, training=True):
        Z = tf.matmul(X, self.W) + self.b
        return self.activation(Z)


class Conv2PoolingLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.first_conv = ConvLayer(id + "_" + "1", M1, M2, activation)
        self.second_conv = ConvLayer(id + "_" + "2", M2, M2, activation)
        self.pooling_layer = PoolingLayer()
    

    def forward(self, X, training=True):
        Z1 = self.first_conv.forward(X)
        Z2 = self.second_conv.forward(Z1)
        return self.pooling_layer.forward(Z2)


class Conv3PoolingLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.first_conv = ConvLayer(id + "_" + "1", M1, M2, activation)
        self.second_conv = ConvLayer(id + "_" + "2", M2, M2, activation)
        self.third_conv = ConvLayer(id + "_" + "3", M2, M2, activation)
        self.pooling_layer = PoolingLayer()

    def forward(self, X, training=True):
        Z1 = self.first_conv.forward(X)
        Z2 = self.second_conv.forward(Z1)
        Z3 = self.third_conv.forward(Z2)
        return self.pooling_layer.forward(Z3)


class Flatten(Layer):
    def forward(self, X, training=True):
        shape = X.get_shape().as_list()
        return tf.reshape(X, (-1, np.prod(shape[1:])))


class ConvBatchLayer(Layer):
    def __init__(self, id, M1, M2, activation):
        self.activation = activation
        self.id = id
        self.M2 = M2
        self.init = False
        filter_shape = [3, 3, M1, M2]

        W_init = self.__init_weight(filter_shape)
        b_init = tf.zeros(M2, dtype=tf.float32)
        self.W = tf.Variable(W_init, name="W" + id)
        self.b = tf.Variable(b_init, name="b" + id)
        

    def forward(self, X, training=True):
        conv_out = tf.nn.conv2d(X, self.W, [1,1,1,1], padding='SAME')

        if not self.init:
            conv_shape = conv_out.get_shape().as_list()
            self.gamma = tf.Variable(tf.ones(conv_shape[1:]), dtype=tf.float32)
            self.global_mean = tf.Variable(tf.zeros(conv_shape[1:]), dtype=tf.float32, trainable=False)
            self.global_var = tf.Variable(tf.zeros(conv_shape[1:]), dtype=tf.float32, trainable=False)

        if training:
            batch_mean, batch_var = tf.nn.moments(conv_out, [0])
            update_global_mean = tf.assign(
                self.global_mean,
                self.global_mean * 0.9 + batch_mean * (0.1)
            )
            update_global_var = tf.assign(
                self.global_var,
                self.global_var * 0.9 + batch_var * (0.1)
            )
            
            with tf.control_dependencies([update_global_mean, update_global_var]):
                out = tf.nn.batch_normalization(
                    conv_out, 
                    batch_mean, 
                    batch_var,
                    self.b,
                    self.gamma, 
                    1e-4
                )

        else:
            out = tf.nn.batch_normalization(
                conv_out, 
                self.global_mean,
                self.global_var,
                self.b, 
                self.gamma,
                1e-4
            )
    

        return self.activation(out)

    def __init_weight(self, shape):
        return tf.random.normal(shape, dtype=tf.float32) / np.sqrt(2.0 / np.prod(shape[:-1])).astype(np.float32)


class Conv2BatchPoolingLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.first_conv = ConvLayer(id + "_" + "1", M1, M2, activation)
        self.second_conv = ConvBatchLayer(id + "_" + "2", M2, M2, activation)
        self.pooling_layer = PoolingLayer()
    

    def forward(self, X, training = True):
        Z1 = self.first_conv.forward(X)
        Z2 = self.second_conv.forward(Z1, training)
        return self.pooling_layer.forward(Z2)


class Conv2Batch2PoolingLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.first_conv = ConvBatchLayer(id + "_" + "1", M1, M2, activation)
        self.second_conv = ConvBatchLayer(id + "_" + "2", M2, M2, activation)
        self.pooling_layer = PoolingLayer()
    

    def forward(self, X, training = True):
        Z1 = self.first_conv.forward(X, training)
        Z2 = self.second_conv.forward(Z1, training)
        return self.pooling_layer.forward(Z2)


class VanillaBatchLayer(Layer):
    def __init__(self, id, M1, M2, activation=tf.nn.relu):
        self.activation = activation
        self.id = id

        W_init = tf.random.normal((M1, M2), dtype=tf.float32)
        
        self.W = tf.Variable(W_init, name="W" + id)
        self.b = tf.Variable(tf.zeros((M2,), dtype=tf.float32), name="b" + id)
        self.gamma = tf.Variable(tf.ones((M2,), dtype=tf.float32), name="gamma" + id)

        self.global_mean = tf.Variable(tf.zeros((M2,), dtype=tf.float32), trainable=False)
        self.global_var = tf.Variable(tf.zeros((M2,), dtype=tf.float32), trainable=False)


    def forward(self, X, training):
        Z = tf.matmul(X, self.W)

        if training:
            batch_mean, batch_var = tf.nn.moments(Z, [0])
            update_global_mean = tf.assign(
                self.global_mean,
                self.global_mean * 0.9 + batch_mean * 0.1
            )
            update_global_var = tf.assign(
                self.global_var,
                self.global_var * 0.9 + batch_var * 0.1
            )

            with tf.control_dependencies([update_global_mean, update_global_var]):
                out = tf.nn.batch_normalization(
                    Z,
                    batch_mean,
                    batch_var,
                    self.b,
                    self.gamma,
                    1e-4
                )
        
        else:
            out = tf.nn.batch_normalization(
                Z,
                self.global_mean,
                self.global_var,
                self.b,
                self.gamma,
                1e-4
            )
        
        return self.activation(out)


class DropoutLayer(Layer):
    def __init__(self, rate):
        assert 0 <= rate <= 1
        self.rate = rate

    def forward(self, X, training=True):
        return tf.nn.dropout(X, keep_prob=1-self.rate)