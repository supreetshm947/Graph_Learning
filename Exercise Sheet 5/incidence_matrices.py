import scipy as sp
import numpy as np
import tensorflow as tf


def generate_matrices(graph, max_nodes, max_edges):
    """
    A function to create the directed incidence matrices (I_in & I_out) of a given graph
    :param graph: A networkx graph
    :return: padded directed incidence matrices I_in & I_out
    """
    G = graph.to_directed()  # Converting to a directed graph
    H = list(G.edges)  # Directed edge set
    edge_length = len(H)
    node_length = len(G.nodes)

    degrees = np.array([val for (node, val) in graph.degree()], dtype=float)  # Degree of every node in the graph

    edges = np.arange(edge_length, dtype=int)  # using edges as the rows of COO-format incidence matrices
    node_out = np.zeros((edge_length,), dtype=int)  # using as the columns of COO-format I_out matrix
    node_in = np.zeros((edge_length,), dtype=int)  # using as the columns of COO-format I_in matrix

    # Data for COO-format matrix as 1s
    data_out = np.ones((edge_length,), dtype=int)
    data_in = np.ones((edge_length,), dtype=float)

    for i in range(len(H)):
        node_out[i], node_in[i] = H[i][0], H[i][1]
        # data_in[i] = data_in[i] / degrees[node_in[i]]  # Dividing by degree of node for mean aggregation
        data_in[i] = data_in[i]

    coo_out = tf.convert_to_tensor(sp.sparse.coo_matrix((data_out, (edges, node_out)),
                                                        shape=(edge_length, node_length)).toarray(), dtype=float)
    coo_in = tf.convert_to_tensor(sp.sparse.coo_matrix((data_in, (edges, node_in)),
                                                       shape=(edge_length, node_length)).toarray(), dtype=float)

    paddings = [[0, max_edges - edge_length], [0, max_nodes - node_length]]  # Padding dimensions for incidence matrix
    incidence_out = tf.pad(coo_out, paddings, 'CONSTANT')
    incidence_in = tf.pad(coo_in, paddings, 'CONSTANT')

    return incidence_out, incidence_in
