import tensorflow as tf
import Pooling_layer as pl


class Crawl(tf.keras.layers.Layer):

    def __init__(self, units, r):
        super(Crawl, self).__init__()

        self.units = units
        self.r = r

        self.conv1 = tf.keras.layers.Conv1D(self.units, self.r + 1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(self.units, self.r + 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.node_pool = pl.NodePool(self.units, self.r)
        self.update = tf.keras.layers.Dense(self.units, trainable=True, activation='relu',
                                            kernel_initializer=tf.keras.initializers.glorot_normal)

    def call(self, inputs, *args, **kwargs):

        inputs = self.conv1(inputs)
        print(inputs.shape)
        inputs = self.bn1(inputs)
        print(inputs.shape)
        inputs = self.relu1(inputs)
        print(inputs.shape)
        inputs = self.conv2(inputs)
        print(inputs.shape)
        inputs = self.bn2(inputs)
        print(inputs.shape)
        inputs = self.relu2(inputs)

        pool = self.node_pool(inputs, *args)

        return self.update(pool)

