import tensorflow as tf
from tensorflow import keras
import numpy as np
import data_utils
import gcn as gc
import utils as us

def createModel(g, epochs):
    """
    Create a GCN Model for node classification task
    :param g: A networkx graph
    :param epochs: Training iterations for model
    :return: Trained GCN Model
    """

    # Get adjacency matrix for graph
    adj = data_utils.get_adjacency_matrix(g)

    # Get the normalized adjacency matrix
    adj_norm = us.normalize_adj(adj)

    # Get features as node_attributes of training graph
    X = data_utils.get_node_attributes(g)

    # Get the target node labels of training graph
    Y = data_utils.get_node_labels(g) - 1  # Making the labels start from 0

    # Setting normalized adjacency matrix (a_tilde) as class level var
    gc.GraphConv.a_tilde = us.sp_matrix_to_sp_tensor(adj_norm)

    # Creating our 2-layer GCN Model for Node Classification Task
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=np.shape(X)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(gc.GraphConv(32, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(gc.GraphConv(len(np.unique(Y)), activation="softmax"))

    # Compiling the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    model.fit(tf.expand_dims(X, axis=0), tf.expand_dims(Y, axis=0), epochs=epochs, batch_size=1)

    return model

def test_model(model, X, Y):
    """
    Evaluating GCN model
    :param model: Trained GCN model
    :param X: Node attributes of test graph
    :param Y: Node labels of test graph
    :return: Test accuracy scores
    """
    score = model.evaluate(tf.expand_dims(X, axis=0), tf.expand_dims(Y, axis=0), verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')