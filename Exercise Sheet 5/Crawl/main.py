import random
import numpy as np
import pickle
import tensorflow as tf
import GlobalPoolingLayer as gpl
import data_utils as us

# Loading datasets
with open('../datasets/HIV/HIV_Val/data.pkl', 'rb') as f:
    graphs = pickle.load(f)

# graphs.pop(3)

# G = graphs[0]
l = 50
s = 4
r = 2


def random_walk(G, start_node, walk_length):
    """

    :param G:
    :param start_node:
    :param walk_length:
    :return:
    """
    walk = [start_node]

    for w in range(walk_length):

        cur = walk[-1]
        cur_nbrs = sorted(list(G.neighbors(cur)))

        if len(cur_nbrs) > 0:

            if len(walk) == 1:
                walk.extend(random.sample(cur_nbrs, 1))

            else:

                if G.degree(cur) == 1:
                    walk.extend(random.sample(cur_nbrs, 1))

                else:
                    prev = walk[-2]
                    cur_nbrs.remove(prev)
                    walk.extend(random.sample(cur_nbrs, 1))

        else:
            break

    return walk


def all_walks(G, walk_length):
    """
    Performing random walks for all nodes of graph G
    :param G: A networkx graph
    :param walk_length: No. of traversed nodes in a random walk
    :return: A list of random walks for all nodes of graph
    """

    nodes = [n for n in G.nodes()]
    walks = []

    random.shuffle(nodes)
    for node in nodes:
        walks.append(random_walk(G, node, walk_length))

    return sorted(walks)


# # Add virtual node
# # G.add_node('virtual_node', node_attributes=np.zeros((86,), dtype=float))
# # for i in range(len(G.nodes) - 1):
# #     G.add_edge('virtual_node', i, edge_attributes=np.zeros((6,), dtype=float))
# #
# # print(list(G.neighbors('virtual_node')))
# # print(G.nodes(data=True))
# # print(G.edges(data=True))


def feat(G, walk, s):
    """

    :param G:
    :param walk:
    :param s:
    :return:
    """
    id_feat = []
    con_feat = []
    for i in range(len(walk)):
        node_identity = []
        edge_connectivity = []

        for j in range(s + 1):
            if i - j >= 0 and walk[i - j] == walk[i]:
                node_identity.append(1)
            else:
                node_identity.append(0)

        node_identity.pop(0)
        id_feat.append(node_identity)

        for j in range(s):
            if i - j >= 1 and G.has_edge(walk[i], walk[i - j - 1]):
                edge_connectivity.append(1)
            else:
                edge_connectivity.append(0)

        edge_connectivity.pop(0)
        con_feat.append(edge_connectivity)

    return np.array(id_feat), np.array(con_feat)


def feature_matrix(G, walk, s):
    """

    :param G:
    :param walk:
    :param s:
    :return:
    """
    d_node = 86
    d_edge = 6
    q = np.zeros((len(walk), d_node + d_edge + s + (s - 1)))
    id_feat, con = feat(G, walk, s)

    for i in range(len(walk)):
        if i == 0:
            q[i] = np.concatenate((G.nodes[walk[i]]['node_attributes'], np.zeros((6,)),
                                   id_feat[i], con[i]), axis=0)

        else:
            q[i] = np.concatenate((G.nodes[walk[i]]['node_attributes'],
                                   G.edges[walk[i], walk[i - 1]]['edge_attributes'],
                                   id_feat[i], con[i]), axis=0)

    return q


def node_features(graphs):
    node_feat =
    for G in graphs:


def center(walks, r, node):
    coo = []

    for i in range(len(G.nodes())):
        for j in range(r + 1, l - r):
            if walks[i][j] == node:
                coo.append((i, j))

    return coo


pos = []
for i in range(len(G.nodes())):
    pos.append(center(all_walks(G, l), 2, i))



class NodePool(tf.keras.layers.Layer):

    def __init__(self, units, r, G, **kwargs):
        super(NodePool, self).__init__(**kwargs)
        self.units = units
        self.r = r
        self.G = G

    def call(self, inputs, *args, **kwargs):

        p = np.zeros((len(self.G.nodes()), self.units), dtype=float)

        for k in range(len(p)):
            t = []
            for l in range(len(args[0][k])):
                i, j = args[0][k][l]
                t.append(inputs[i][j-self.r])

            p[k] = np.mean(np.array(t), axis=0)

        return tf.convert_to_tensor(p)


class Crawl(tf.keras.layers.Layer):

    def __init__(self, units, r, G):
        super(Crawl, self).__init__()

        self.units = units
        self.r = r
        self.G = G

        self.conv1 = tf.keras.layers.Conv1D(self.units, self.r + 1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(self.units, self.r + 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.node_pool = NodePool(self.units, self.r, self.G)
        self.update = tf.keras.layers.Dense(self.units, trainable=True, activation='relu',
                                            kernel_initializer=tf.keras.initializers.glorot_normal)

    def call(self, inputs, *args, **kwargs):

        print(args[0])
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
        inputs = self.relu2(tf.squeeze(inputs))

        pool = self.node_pool(inputs, args[0])

        return self.update(pool)


convv = Crawl(128, r, G)
y = convv(w, pos)
print(y)

