import pickle
import networkx as nx
import incidence_matrices as im
import tensorflow as tf
import mpgnn_layer
import PoolingLayer as pl
import data_utils as us

# Loading datasets
with open('datasets/ZINC_Train/data.pkl', 'rb') as a:
    graphs_zinc_train = pickle.load(a)

with open('datasets/ZINC_Val/data.pkl', 'rb') as b:
    graphs_zinc_val = pickle.load(b)

with open('datasets/ZINC_Test/data.pkl', 'rb') as c:
    graphs_zinc_test = pickle.load(c)

G_train = graphs_zinc_train
G_val = graphs_zinc_val
G_test = graphs_zinc_test

# Loading output graph labels for each dataset and converting to tensors
labels_train, labels_val, labels_test = [], [], []

for g in graphs_zinc_train:
    labels_train.extend(us.get_graph_label(g))
labels_train = tf.convert_to_tensor(labels_train)

for g in graphs_zinc_val:
    labels_val.extend(us.get_graph_label(g))
labels_val = tf.convert_to_tensor(labels_val)

for g in graphs_zinc_test:
    labels_test.extend(us.get_graph_label(g))
labels_test = tf.convert_to_tensor(labels_test)


def preprocess_data(G):
    """
    Function to pad the input features (incidence matrices, edge & node features) to achieve uniform size
    over all graphs of a dataset.
    :param G: A set of networkx graphs
    :return: the input features (incidence matrices, edge & node features) converted to tensors
    """

    # Initializing empty input features
    global incidence_in, incidence_out
    incidence_matrices = []
    features_node = []
    features_edge = []
    edge_labels = []
    node_labels = []

    # Computing the max dimension of edges and nodes over all the graphs in the given dataset G
    max_nodes, max_edges = list(map(lambda x: max(x), zip(*[(len(g.nodes),
                                                             len(g.to_directed().edges)) for g in G])))

    for graphs in G:
        # Generating padded incidence matrices from with dimensions (max_edges x max_nodes)
        incidence_matrices.append(im.generate_matrices(graphs, max_nodes, max_edges))
        incidence_out, incidence_in = list(zip(*incidence_matrices))

        # Putting node and edge labels of all the graphs into one list
        node_labels.append([n for n in nx.get_node_attributes(graphs, "node_label").values()])
        edge_labels.append([graphs.get_edge_data(e[0],
                                                 e[1])["edge_label"] for e in graphs.to_directed().edges()])

    # Computing the max dimension of edge and node labels for padding over all graphs
    max_edge_label = max(max(edge_labels)) + 1
    max_node_label = max(max(node_labels)) + 7

    for i in range(len(edge_labels)):
        # Converting edge labels to one-hot vectors to be used as edge features
        edge_labels[i] = tf.convert_to_tensor(tf.keras.utils.to_categorical(edge_labels[i]))

        # Padding dimensions for edge features of each graph
        paddings_edge = [[0, max_edges - edge_labels[i].get_shape()[0]],
                         [0, max_edge_label - edge_labels[i].get_shape()[1]]]

        # Padding the edge features of each graph with 0 and appending to a list
        features_edge.append(tf.pad(edge_labels[i], paddings_edge, "CONSTANT"))

        # Converting node labels to one-hot vectors to be used as initial node features
        node_labels[i] = tf.convert_to_tensor(tf.keras.utils.to_categorical(node_labels[i]))

        # Padding dimensions for node features of each graph
        paddings_node = [[0, max_nodes - node_labels[i].get_shape()[0]],
                         [0, max_node_label - node_labels[i].get_shape()[1]]]

        # Padding the node features of each graph with 0 and appending to a list
        features_node.append(tf.pad(node_labels[i], paddings_node, "CONSTANT"))

    return tf.convert_to_tensor(incidence_out), tf.convert_to_tensor(incidence_in), \
           tf.convert_to_tensor(features_edge), tf.convert_to_tensor(features_node)


# Generating the input features for training dataset
inc_out_train, inc_in_train, edge_feat_train, node_feat_train = preprocess_data(G_train)
inc_out_val, inc_in_val, edge_feat_val, node_feat_val = preprocess_data(G_val)
inc_out_test, inc_in_test, edge_feat_test, node_feat_test = preprocess_data(G_test)

# Padding dimensions to achieve uniform size to achieve a uniform size across the whole dataset (train, val & test).
pad_node_train = [[0, 0], [0, 1]]  # Padding for node features of training dataset
pad_i_out_train = [[0, 2], [0, 0]]  # Padding for incidence-out matrices of training dataset
pad_edge_train = [[0, 2], [0, 0]]  # Padding for edge features of training dataset
pad_iin_train = [[0, 2], [0, 0]]  # Padding for incidence-in matrices of training dataset
pad_node_val = [[0, 1], [0, 2]]  # Padding for node features of validation dataset
pad_i_out_val = [[0, 4], [0, 1]]  # Padding for incidence-out matrices of validation dataset
pad_edge_val = [[0, 4], [0, 0]]  # Padding for edge features of validation dataset
pad_iin_val = [[0, 4], [0, 1]]  # Padding for incidence-in matrices of validation dataset

n_train_up, i_out_train_up, e_train_up, iin_train_up = [], [], [], []
n_val_up, i_out_val_up, e_val_up, iin_val_up = [], [], [], []


def pad_across_dataset(feature, padding, updated):
    """
    Function to pad features to achieve uniform size across whole dataset
    :param feature: Input feature to be padded
    :param padding: Padding dimensions
    :param updated: List of all padded features
    :return: tensor of all padded features
    """
    for i in range(feature.shape[0]):
        updated.append(tf.pad(feature[i], padding, "CONSTANT"))

    return tf.convert_to_tensor(updated)


node_feat_train = pad_across_dataset(node_feat_train, pad_node_train, n_train_up)
inc_out_train = pad_across_dataset(inc_out_train, pad_i_out_train, i_out_train_up)
edge_feat_train = pad_across_dataset(edge_feat_train, pad_edge_train, e_train_up)
inc_in_train = pad_across_dataset(inc_in_train, pad_iin_train, iin_train_up)

node_feat_val = pad_across_dataset(node_feat_val, pad_node_val, n_val_up)
inc_out_val = pad_across_dataset(inc_out_val, pad_i_out_val, i_out_val_up)
edge_feat_val = pad_across_dataset(edge_feat_val, pad_edge_val, e_val_up)
inc_in_val = pad_across_dataset(inc_in_val, pad_iin_val, iin_val_up)


class MPGNN_Model(tf.keras.models.Model):
    """
    A keras model for implementing a graph regression task using Message-Passing Graph Neural Network Layers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = mpgnn_layer.GNN(64, skip=True)  # First MPGNN Layer with units=16 and using skip connection
        self.layer2 = mpgnn_layer.GNN(64, skip=True)  # Second MPGNN Layer with units=8 and possibly using skip connection
        self.layer3 = mpgnn_layer.GNN(64)  # Third MPGNN Layer with units=4
        self.pool = pl.SumPooling()  # A custom sum-pooling layer
        self.dense1 = tf.keras.layers.Dense(1, use_bias=True, activation=None)  # Final Dense layer with a single unit

    def call(self, inputs, training=None, mask=None):
        inputs_second, first_layer = self.layer1(inputs)
        inputs_third, second_layer = self.layer2(
            inputs_second)  # Second layer gets updated input node features with possible skip-connection
        _, third_layer = self.layer3(
            inputs_third)  # Third layer gets updated input node features with possible skip-connection
        pool_layer = self.pool(third_layer)
        dense_layer1 = self.dense1(pool_layer)

        return dense_layer1


<<<<<<< HEAD
print(node_feat_train.shape)
#
# model = MPGNN_Model()
#
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss=tf.keras.losses.mean_absolute_error,
#               metrics=['mean_absolute_error'])
#
# history = model.fit(x=[node_feat_train, inc_out_train, edge_feat_train, inc_in_train], y=labels_train,
#                     batch_size=100, epochs=500,
#                     validation_data=([node_feat_val, inc_out_val, edge_feat_val, inc_in_val], labels_val))
#
# results = model.evaluate([node_feat_test, inc_out_test, edge_feat_test, inc_in_test],
#                          labels_test, batch_size=10)
=======
model = MPGNN_Model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.mean_absolute_error,
              metrics=['mean_absolute_error'])
model.run_eagerly = True
history = model.fit(x=[node_feat_train, inc_out_train, edge_feat_train, inc_in_train], y=labels_train,
                    batch_size=100, epochs=500,
                    validation_data=([node_feat_val, inc_out_val, edge_feat_val, inc_in_val], labels_val))

results = model.evaluate([node_feat_test, inc_out_test, edge_feat_test, inc_in_test],
                         labels_test, batch_size=10)
>>>>>>> 84ffc6df8dd29589c9636e1c43a85852834852d0
