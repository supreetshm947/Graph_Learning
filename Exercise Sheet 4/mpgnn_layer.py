import tensorflow as tf
import residual_layer as rl


class GNN(tf.keras.layers.Layer):
    """
    Custom Graph Neural Network Layer to implement Message Passing Graph Neural Network
    """

    def __init__(self, units, skip=False):
        super(GNN, self).__init__()

        self.units = units  # Defining no. of units of layer
        self.skip = skip  # Parameter for deciding skip connection

        # Layer for aggregating the messages (M^(l))
        self.aggregate = tf.keras.layers.Dense(self.units, use_bias=False, activation='relu', trainable=True,
                                               kernel_initializer=tf.keras.initializers.glorot_normal)

        # Layer for updating the input node features (U^(l))
        self.update = tf.keras.layers.Dense(self.units, use_bias=False, activation='relu', trainable=True,
                                            kernel_initializer=tf.keras.initializers.glorot_normal)

        # Layer for computing the skip connections
        self.res = rl.Residual(self.units)

    def call(self, inputs, *args, **kwargs):
        prev = tf.matmul(inputs[1], inputs[0])  # I_out . X^(l-1)
        prev = tf.concat([inputs[2], prev], axis=-1)  # [F^(E), I_out . X^(l-1)]
        messages = self.aggregate(prev)  # M^(l) ([F^(E), I_out . X^(l-1)])
        cur = tf.matmul(tf.transpose(inputs[3], perm=[0, 2, 1]), messages)  # (I_in)' . S^(l)
        cur = tf.concat([inputs[0], cur], axis=-1)  # [X^(l-1), tr(I_in) . S^(l)]
        updated = self.update(cur)  # U^(l) ([X^(l-1), tr(I_in) . S^(l)])

        if self.skip:
            next = self.res([inputs[0], updated])  # Implementing skip connection
            next_inputs = [next, inputs[1], inputs[2], inputs[3]]  # Updating the inputs for next layer with output from skip connection

            return next_inputs, updated

        next_inputs = [updated, inputs[1], inputs[2], inputs[3]]  # Simply updating the input node features without skip connections

        return next_inputs, updated
