import residual_layer as rl
import incidence_matrices as im
import data_utils
import tensorflow as tf
import PoolingLayer as pl
import argparse
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras import activations
import tensorflow.keras.metrics


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
            next_inputs = [next, inputs[1], inputs[2],
                           inputs[3]]  # Updating the inputs for next layer with output from skip connection

            return next_inputs, updated

        next_inputs = [updated, inputs[1], inputs[2],
                       inputs[3]]  # Simply updating the input node features without skip connections

        return next_inputs, updated


class MPGNN_Model(tf.keras.models.Model):
    """
    A keras model for implementing a graph regression task using Message-Passing Graph Neural Network Layers
    """

    def __init__(self, num_of_layers, units, skip, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layers = []
        for i in range(num_of_layers):
            self.hidden_layers.append(GNN(units, skip=skip))
        self.layer_n = GNN(units)
        self.pool = pl.SumPooling()
        self.dense1 = tf.keras.layers.Dense(1, use_bias=True,
                                            activation=activations.sigmoid)  # Final Dense layer with a single unit

    def call(self, inputs):
        for layer in self.hidden_layers:
            inputs = layer(inputs)[0]
        out, updated_labels = self.layer_n(inputs)
        out = self.pool(updated_labels)
        dense_layer1 = self.dense1(out)
        return dense_layer1


def pad(matrix, max_len):
    """
        Function to pad features
        :param matrix: Input feature to be padded
        :param max_len: the length of maximum feature
        :return: tensor of padded features
    """
    return tf.pad(matrix, [[0, max_len - tf.shape(matrix)[0]], [0, 0]], "CONSTANT")


def preprocess_data(all_data):
    """
        Function to create input features, i.e., Node Attributes, Incidence Matrix and Edge Attributes
        :param all_data: input data, a list of graphs
        :return: input data for the model and the output labels
    """
    max_nodes, max_edges = list(
        map(lambda x: max(x),
            zip(*[(len(graph.nodes), len(graph.to_directed().edges)) for data in all_data.values() for graph in
                  data])))
    input_data = {}
    for key in all_data.keys():
        data = all_data.get(key)
        incidence_matrices = []
        node_attributes = []
        edge_attributes = []
        graph_labels = []
        for graph in data:
            incidence_matrices.append(im.generate_matrices(graph, max_nodes, max_edges))
            node_attributes.append(pad(data_utils.get_node_attributes(graph), max_nodes))
            edge_attributes.append(
                pad([graph.get_edge_data(e[0], e[1])['edge_attributes'] for e in graph.to_directed().edges()],
                    max_edges))
            graph_labels.append(data_utils.get_graph_label(graph))
        con = tf.convert_to_tensor
        incidence_out, incidence_in = list(zip(*incidence_matrices))
        input_data.update(
            {key: [con(node_attributes), con(incidence_out), con(edge_attributes), con(incidence_in),
                   con(graph_labels)]})
    return input_data


def train_model(all_data, num_of_layers, layer_units, skip_resnet, learning_rate, epochs, batch_size):
    """
        Function to train the model
        :param all_data: Dict of input data and labels
        :param num_of_layers: number of layers for the model
        :param layer_units: unit dimension length for hidden layers
        :param skip_resnet: flag for the residual network in model
        :param learning_rate: learning rate to train the model
        :param epochs: number of epochs to train the model
        :param batch_size: batch_size of input data to train the model
        :return: trained model
    """
    model = MPGNN_Model(num_of_layers=num_of_layers, units=layer_units, skip=skip_resnet)
    train_data = all_data.get("train")
    val_data = all_data.get("val")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.mean_absolute_error,
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.run_eagerly = True

    model.fit(x=[train_data[0], train_data[1], train_data[2], train_data[3]], y=train_data[4],
              batch_size=batch_size, epochs=epochs,
              validation_data=([val_data[0], val_data[1], val_data[2], val_data[3]], val_data[4]))
    auc_train = roc_auc_score(train_data[4],
                              model.predict([train_data[0], train_data[1], train_data[2], train_data[3]]),
                              average='macro')
    print("AUC Train: %f Train Accuracy: %f" % (auc_train))
    return model


def test_model(model, data):
    """
        Function to test the model
        :param model: model
        :param data: dict of input data and labels
    """
    if data.get('test'):
        test_data = data.get('test')
        return model.evaluate([test_data[0], test_data[1], test_data[2], test_data[3]],
                              test_data[4], batch_size=100)


with open('datasets/HIV/HIV_Train/data.pkl', 'rb') as a:
    graphs_hiv_train = pickle.load(a)

with open('datasets/HIV/HIV_Val/data.pkl', 'rb') as a:
    graphs_hiv_val = pickle.load(a)

parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument('num_of_layers', type=int, default=5, nargs="?")
parser.add_argument('layer_units', type=int, default=128, nargs="?")
parser.add_argument('skip_resnet', type=bool, default=True, nargs="?")
parser.add_argument('learning_rate', type=float, default=1e-3, nargs="?")
parser.add_argument('epochs', type=int, default=1, nargs="?")
parser.add_argument('batch_size', type=int, default=100, nargs="?")
args = parser.parse_args()

all_data = preprocess_data({"train": graphs_hiv_train, "val": graphs_hiv_val, "test": graphs_hiv_val})

model = train_model(all_data, args.num_of_layers, args.layer_units, args.skip_resnet,
                    args.learning_rate, args.epochs, args.batch_size)

print(test_model(model=model, data=all_data))
