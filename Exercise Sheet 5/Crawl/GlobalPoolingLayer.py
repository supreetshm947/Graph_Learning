from tensorflow.keras.layers import Layer
import tensorflow as tf


class SumPooling(Layer):

    def __init__(self, **kwargs):
        super(SumPooling, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        x = tf.math.reduce_sum(tf.transpose(x, perm=[0, 2, 1]), axis=2)

        return tf.squeeze(x)
