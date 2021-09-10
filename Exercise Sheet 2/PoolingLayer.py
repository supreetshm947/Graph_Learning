from tensorflow.keras.layers import Layer
import tensorflow as tf

class PoolingLayer(Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(PoolingLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.reshape(tf.math.reduce_sum(x, axis=2), x.shape[0:2])