import random

def prob_edges(G, p, q, prev, cur):
    '''
    Pre-process edges
    :param G: A networkx graph
    :param p: controls the likelihood of immediately revisiting a node in the walk. Low value of p -> BFS-like walk
    :param q: (Low value of q -> DFS-like walk)
    :param prev: v_{i-1} or "source" node
    :param cur: v_{i} or "current" node
    :return: none
    '''
    probs = []
    for next in sorted(list(G.neighbors(cur))):
        weight = G.get_edge_data(cur, next)['weight']
        if next == prev:
            G[cur][next]['weight'] = weight/p  # If next = v_{i}, cur goes back to prev(v_{i-1})

        elif G.has_edge(prev, next):
            G[cur][next]['weight'] = 1  # If v_{i-1} and next node are also connected;
            # then prob of cur -> next = 1

        else:
            G[cur][next]['weight'] = weight/q  # If v_{i-1} and next aren't directly connected;
            # prob of cur -> next = 1/q

        probs.append(G[cur][next]['weight'])

    for next in sorted(list(G.neighbors(cur))):
        G[cur][next]['weight'] = float(G[cur][next]['weight'])/sum(probs)  # Normalizing probabilities/weights


def max_sampling(G, cur, max_prob):
    '''
    Sampling next nodes by taking max weights
    :param cur:
    :param max_prob:
    :return: the node with max weight
    '''
    sampled_nodes = []
    # Gets the weights of cur -> next;
    for k, v in G[cur].items():
        for k1, v1 in G[cur][k].items():
            if G[cur][k][k1] == max_prob[0]:
                sampled_nodes.append(k)

    return random.sample(sampled_nodes, 1)  # Randomly sampling from nodes with max weights


def random_walk(G, walk_length, start_node, p, q):
    """
    Performing biased random walk for given starting node and return parameters p & q
    :param G: A networkx graph
    :param walk_length: The length of random walk
    :param start_node: Starting node for performing random walk from
    :param p: Low value of p -> BFS-like walk
    :param q: Low value of q -> DFS-like walk
    :return: A list of traversed nodes from given starting node
    """
    walk = [start_node]

    for w in range(walk_length):
        cur = walk[-1]  # cur is the last obtained node from previous step
        cur_nbrs = sorted(list(G.neighbors(cur)))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.extend(random.sample(cur_nbrs, 1))  # If starting node then sample any next node (same weights)

            else:
                prev = walk[-2]
                prob_edges(G, p, q, prev, cur)  # Changing edge weights from cur -> next; given prev node
                max_prob = [max([G[cur][v]['weight']
                                 for v in cur_nbrs])]  # Max edge weight from cur -> next
                next = max_sampling(G, cur, max_prob)  # Returns next node with max edge weight
                walk.extend(next)
        else:
            break

    return walk

def all_walks(G, walk_length, num_walks, p, q):
    """
    Performing random walks for all nodes of graph G
    :param G: A networkx graph
    :param walk_length: No. of traversed nodes in a random walk
    :param num_walks: No. of random walks per node
    :param p: Low value of p -> BFS-like walk
    :param q: Low value of q -> DFS-like walk
    :return: A list of random walks for all nodes of graph
    """
    # Initializing graph edges with uniform weights
    for u in G.nodes():
        for v in G.nodes():
            if G.has_edge(u, v):
                G[u][v]['weight'] = 1.0

    nodes = [n for n in G.nodes()]
    walks = []

    for walk_iter in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, walk_length, node, p, q))

    return sorted(walks)

def negative_sampling(G, walks, walk_length):
    """
    Sampling complement of walks \in G.nodes(), where len({walk}^\complement) = len(walk)
    :param G: A networkx graph
    :param walks: A set of random walks
    :param walk_length: No. of traversed nodes in a random walk
    :return: A set of complement of all walk \in walks
    """
    anti_walks = []
    res = []

    for n in range(len(walks)):
        res.append(list(set(G.nodes()) - set(walks[n])))
        anti_walks.append(random.sample(res[n], walk_length))

    return anti_walks



