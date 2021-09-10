import tensorflow as tf
from tensorflow import keras
import numpy as np
import pydot
import data_utils
import pickle
import gcn as gc
import utils as us
import os
import PoolingLayer as pl

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

with open('datasets/ENZYMES/data.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Getting the combined, padded adjacency matrices of all graphs in dataset
A = data_utils.get_padded_adjacency(graphs)

# l2 normalizing the node attributes of graphs in ENZYMES dataset (NA for NCI1)
data_norm = us.l2_normalise(data_utils.get_padded_node_attributes(graphs))

# The final input vectors of each graph are the combination of node attributes and one-hot node labels
X = us.append_node_labels(data_norm, data_utils.get_padded_node_labels(graphs))

# Normalize the adjacency matrices of the graphs
A_tilde = us.normalise_padded_adjaceny(A)

# Setting class var
gc.GraphConv.a_tilde = A_tilde

# Get labels of each graph
Y = us.get_graph_labels(graphs)
print(Y.shape)
# Creating our GCN Model for Graph Classification task
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=np.shape(X)))
model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(X)))
model.add(gc.GraphConv(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.4))
model.add(gc.GraphConv(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.4))
model.add(gc.GraphConv(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.4))
model.add(gc.GraphConv(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.4))
model.add(gc.GraphConv(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(pl.PoolingLayer())
model.add(tf.keras.layers.Dense(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(6, activation="softmax"))


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

keras.ut.plot_model(model, "./Model architecture/GCN-ENZYMES.png", show_shapes=True)
Y = Y-1
model.fit(tf.expand_dims(X, axis=0), tf.expand_dims(Y, axis=0), epochs=50, batch_size=None, verbose=2)







