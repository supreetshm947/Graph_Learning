import tensorflow as tf


class Residual(tf.keras.layers.Layer):
    """
    A keras custom layer to implement skip-connections between layers
    """
    def __init__(self, units):
        super(Residual, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer=tf.keras.initializers.glorot_normal,
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs[0], self.w) + inputs[1]  # X^(l-1).W + U^(l) (X^(l-1), ..)
