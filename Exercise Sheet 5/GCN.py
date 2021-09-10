from tensorflow.python.keras import activations
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow import keras
import data_utils
import numpy as np
import pickle
import argparse
import residual_layer
import Utils as ut

class GraphConv(tf.keras.layers.Layer):
    """
        Custom Graph Convolutional Network Layer
    """
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConv, self).__init__()
        self.units = units
        self.activation_str = activation
        self.activation = activations.get(activation)

    def build(self, input_shape):
        w_init = initializers.glorot_normal()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True)

    def call(self, inp, **kwargs):
        x = tf.matmul(inp, self.w)
        return self.activation(x)


class GCNModel(keras.models.Model):
    """
        A keras model for implementing a graph convolutional network
    """
    def __init__(self, hidden_layer_dim, unique_label, num_of_layers, dropout_rate, *args, **kwargs):
        super(GCNModel, self).__init__()
        self.layerss = []
        for i in range(num_of_layers):
            self.layerss.append(
                [GraphConv(hidden_layer_dim, activation="relu"), residual_layer.Residual(hidden_layer_dim),
                 tf.keras.layers.Dropout(dropout_rate)])
        self.outlayer = GraphConv(unique_label, activation="softmax")

    def call(self, inp, *args, **kwargs):
        adj, X = inp
        for layer, res, dropout in self.layerss:
            inp = tf.matmul(adj, X)
            X = layer(inp)
            X = res([inp, X])
            X = dropout(X)
        inp = tf.matmul(adj, X)
        return self.outlayer(inp)

def preprocess_data(data):
    """
        Function to create input and output data for feeding into model
        :param data: input data
        :return: input data for the model and the output labels
    """
    adj = data_utils.get_adjacency_matrix(data)
    adj_norm = ut.normalize_adj(adj)

    X = data_utils.get_node_attributes(data)
    Y = data_utils.get_node_labels(data)

    inputs = []
    inputs.append(ut.sp_matrix_to_sp_tensor(adj_norm))
    inputs.append(tf.convert_to_tensor(X))

    return inputs, Y


parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument('num_of_layers', type=int, default=5, nargs="?")
parser.add_argument('hidden_dim', type=int, default=40, nargs="?")
parser.add_argument('dropout_rate', type=float, default=.3, nargs="?")
parser.add_argument('learning_rate', type=float, default=0.001, nargs="?")
parser.add_argument('epochs', type=int, default=500, nargs="?")
args = parser.parse_args()

with open('datasets/Amazon/Amazon_Train/data.pkl', 'rb') as f:
    graphs_train = pickle.load(f)[0]

inputs_train, Y_train = preprocess_data(graphs_train)

with open('datasets/Amazon/Amazon_Val/data.pkl', 'rb') as f:
    graphs_validation = pickle.load(f)[0]

inputs_val, Y_val = preprocess_data(graphs_validation)

gcn_model = GCNModel(input_shape=inputs_train[1].shape, unique_label=len(np.unique(Y_train)), num_of_layers=args.num_of_layers,
                     hidden_layer_dim=args.hidden_dim, dropout_rate=args.dropout_rate)
gcn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['mean_absolute_error'])
gcn_model.run_eagerly = True
gcn_model.fit(inputs_train, Y_train, batch_size=len(graphs_train.nodes), epochs=args.epochs,
              validation_data=(inputs_val, Y_val), validation_batch_size=len(graphs_validation.nodes))

gcn_model.summary()
