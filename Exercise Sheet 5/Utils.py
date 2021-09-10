import numpy as np

def get_padded_(graphs, func):
    """
    Computes a 3D Tensor A of shape (k,n,n) that stacks all adjacency matrices.
    Here, k = |graphs|, n = max(|V|) and A[i,:,:] is the padded adjacency matrix of the i-th graph.
    :param graphs: A list of networkx graphs
    :return: Numpy array A
    """
    max_size = np.max([g.func for g in graphs])
    A_list = [get_adjacency_matrix(g) for g in graphs]
    A_padded = [np.pad(A, [0, max_size-A.shape[0]]) for A in A_list]

    return np.float32(A_padded)