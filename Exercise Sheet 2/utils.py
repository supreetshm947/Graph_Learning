import numpy as np
import scipy.sparse as sp
import data_utils
import tensorflow as tf


def normalize_adj(adj):
    """
    Symmetrically normalizes the adjacency matrix (A)
    :param adj: Adjacency Matrix
    :return: Normalized adjacency matrix ~{A} = D^{-0.5} * ~{A} * D^{-0.5} in COO format
    """
    # Adding self loops (A+I)
    adj_self = adj + sp.eye(adj.shape[0])
    adj_self = sp.coo_matrix(adj_self)
    # Degree matrix to the power of -1/2 (D^{-0.5})
    D_t = np.power(np.array(adj_self.sum(1)), -0.5).flatten()
    D_t[np.isinf(D_t)] = 0.
    D_t = sp.diags(D_t)

    return D_t.dot(adj_self).dot(D_t).tocoo()

def l2_normalise(data):
    """
    L2 normalizing the node attributes of ENZYMES dataset
    :param data: 3D Tensor with shape (k, n, a) that stacks the node attributes of all graphs.
    Here, k = |graphs|, n = max(|V|) and a is the length of the attribute vectors.
    :return: 3D Tensor with L2 normalized padded node attributes of all graphs
    """
    data_norm = np.empty(np.shape(data))
    for k in range(np.shape(data)[0]):
        l2norm = np.linalg.norm(data[k,:,:], axis=1, ord=2)
        data_norm[k,:,:] = data[k,:,:] / l2norm.reshape(np.shape(data)[1],1)
        data_norm[np.isnan(data_norm)] = 0.

    return data_norm

def append_node_labels(data_norm, y):
    """
    Concatenating the node attributes of graphs with node labels
    :param data_norm: 3D Tensor with L2 normalized padded node attributes of all graphs.
    :param y: 3D Tensor X with shape (k, n, l) that stacks the node labels of all graphs.
    Here,l is the number of distinct node labels.
    :return: 3D Tensor of stacked feature vectors of all graphs with shape (k, n, a+l)
    """
    data = np.empty([600,126,21])
    for i in range(np.shape(data_norm)[0]):
        for j in range(np.shape(data_norm)[1]):
            data[i,j,:] = np.concatenate((data_norm[i,j,:],y[i,j,:]), axis=None)

    return np.float32(data)

def normalise_padded_adjaceny(A):
    """
    Symmetrically normalizing the stacked adjacency matrices of all graphs.
    :param A: a 3D Tensor A of shape (k,n,n) that stacks all adjacency matrices.
    Here, A[i,:,:] is the padded adjacency matrix of the i-th graph.
    :return: Normalized padded adjacency matrices ~{A} = D^{-0.5} * ~{A} * D^{-0.5} in tensor format
    """
    A_tilde = np.empty(np.shape(A))
    for k in range(np.shape(A)[0]):
        A_tilde[k,:,:] = sp_matrix_to_sp_tensor(normalize_adj(A[k,:,:]))

    return np.float32(A_tilde)

def get_graph_labels(graphs):
    """
    :param graphs: A list of networkx graphs
    :return: A list of graph labels of all graphs
    """
    Y = [data_utils.get_graph_label(g) for g in graphs]

    return np.float32(Y)

def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    row, col, values = sp.find(x)
    out = tf.SparseTensor(indices=np.array([row, col]).T, values=values, dense_shape=x.shape)

    return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(out)), tf.float32)
