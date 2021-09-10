import biased_random_walk as brw
import data_utils as utils
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import Word2Vec as w2v
import numpy as np

def classify(G, p, q):
    """
    Function to classify the nodes of graph from their learnt embeddings
    :param G: A networkx graph
    :param p: Controls the likelihood of immediately revisiting a node in the walk. Low value of p -> BFS-like walk
    :param q: Low value of q -> DFS-like walk
    :return: None
    """
    walks = brw.all_walks(G, 4, 5, p, q)

    input = w2v.learn_embeddings(walks,300).wv.vectors  # Node embeddings as input (from Word2Vec module)
    output = utils.get_node_labels(G)

    reg = linear_model.LogisticRegression(max_iter=500)

    scores = cross_val_score(reg, input,
                             output, cv=10)
    print("Params p: %f q: %f Mean Accuracy: %f Standard Deviation: %f" %(p, q, np.mean(scores), np.std(scores)))
