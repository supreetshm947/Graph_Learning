from tensorflow.python.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import tensorflow as tf


# Class for custom Graph Convolutional Layer from keras.layers.Layer
class GraphConv(Layer):

    def __init__(self, units, activation=None, **kwargs):
        super(GraphConv, self).__init__()
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        w_init = initializers.glorot_normal()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        y = tf.matmul(GraphConv.a_tilde, inputs)  # ~{A}*X
        x = tf.matmul(y, self.w)  # ~{A}*X *W

        return self.activation(x)
