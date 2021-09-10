import numpy as np
import networkx as nx
import biased_random_walk as brw
import Word2Vec as w2v
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_evalntrain_edges(G, prop_eval):
    """
       Function to create evaluation and training edge data
       :param G: A networkx graph
       :param prop_eval: proportion of evaluation data
       :return: pos_edge_list_train, pos_edge_list_eval, neg_edge_list_train, neg_edge_list_eval
    """
    n_edges = G.number_of_edges()
    n_eval = int(prop_eval * n_edges)

    rnd = np.random.RandomState()
    # finding positive edges
    edges = G.edges()
    pos_edge_list_eval = []
    pos_edge_list_train = []
    n_count = 0
    rnd_inx = rnd.permutation(n_edges)
    edges = list(edges)
    for eii in rnd_inx:
        edge = edges[eii]
        G.remove_edge(*edge)
        reachable_from_v1 = nx.connected._plain_bfs(G, edge[0])
        if edge[1] not in reachable_from_v1:
            G.add_edge(*edge)
        else:
            pos_edge_list_eval.append(edge)
            n_count += 1
        if n_count >= n_eval:
            break

    pos_edge_list_train = list(filter(lambda x: x not in pos_edge_list_eval, edges))
    # finding negative edges
    non_edges = [e for e in nx.non_edges(G)]
    rnd_inx = rnd.choice(len(non_edges), n_edges, replace=False)
    neg_edge_list_eval = [non_edges[i] for i in rnd_inx[:n_eval]]
    neg_edge_list_train = [non_edges[i] for i in rnd_inx[n_eval:n_edges]]

    return pos_edge_list_train, pos_edge_list_eval, neg_edge_list_train, neg_edge_list_eval


def compute_hadamard_product(node_embeddings, edges):
    """
       Function to compute hadamard product
       :param node_embeddings: node embeddings of the graph
       :param edges: edges for which hadamard product is to be computed
       :return: hadamard_prod
    """
    hadamard_prod = []
    for u, v in edges:
        hadamard_prod.append(node_embeddings[u - 1] * node_embeddings[v - 1])
    return hadamard_prod


def link_prediction(G):
    """
       Function to predict edges of a graph
       :param G: A networkx graph
       :return: hadamard_prod
    """
    # compute the wallks
    walks = brw.all_walks(G, 4, 5, 1, 1)
    # compute the embeddings of the walk
    embeddings = w2v.learn_embeddings(walks, 100).wv.vectors
    aucs_test = []
    aucs_train = []
    accuracies_eval = []
    accuracies_train = []
    for i in range(5):
        G_copy = G.copy()
        # get train, eval negative/positive data
        pos_train, pos_eval, neg_train, neg_eval = compute_evalntrain_edges(G_copy, .2)
        # preparing input and output for training model
        input = compute_hadamard_product(embeddings, pos_train) + compute_hadamard_product(embeddings, neg_train)
        target_output = np.zeros(len(pos_train) + len(neg_train))
        target_output[:len(pos_train)] = 1

        input_eval = compute_hadamard_product(embeddings, pos_eval) + compute_hadamard_product(embeddings, neg_eval)
        output_eval = np.zeros(len(pos_eval) + len(neg_eval))
        output_eval[:len(pos_eval)] = 1

        reg = linear_model.LogisticRegression(max_iter=500)
        reg.fit(input, target_output)

        auc_train = roc_auc_score(target_output, reg.predict_proba(input)[:, 1])

        # Test classifier
        auc_test = roc_auc_score(output_eval, reg.predict_proba(input_eval)[:, 1])
        accuracy_eval = accuracy_score(output_eval, reg.predict(input_eval))
        accuracy_train = accuracy_score(target_output, reg.predict(input))
        aucs_test.append(auc_test)
        aucs_train.append(auc_train)
        accuracies_eval.append(accuracy_eval)
        accuracies_train.append(accuracy_train)
        print("AUC train: %f AUC test: %f Accuracy Train: %f Accuracy Eval: %f for iteration %d"
              % (auc_train, auc_test, accuracy_train, accuracy_eval, i))

    print(
        "Mean AUC train: %f Std. Dev. AUC train: %f Mean AUC test: %f Std. Dev. AUC test: %f Mean Accuracy Train: %f Std. Dev Accuracy Train: %f Mean Accuracy Eval: %f Std. "
        "Dev. Accuracy Eval: %f" % (
        np.mean(aucs_train), np.std(aucs_train), np.mean(aucs_test), np.std(aucs_test), np.mean(accuracies_train),
        np.std(accuracies_train), np.mean(accuracies_eval), np.std(accuracies_eval)))
