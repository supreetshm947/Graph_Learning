from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class NodePool(Layer):

    def __init__(self, units, r, **kwargs):
        super(NodePool, self).__init__(**kwargs)
        self.units = units
        self.r = r

    def call(self, inputs, *args, **kwargs):

        print(args[0])

        p = np.zeros((27, self.units), dtype=float)

        for k in range(len(p)):
            t = []
            for l in range(len(args[0][k])):
                i, j = args[0][k][l]
                t.append(inputs[i][j-self.r])

            p[k] = np.mean(np.array(t), axis=0)

        return tf.convert_to_tensor(p)
