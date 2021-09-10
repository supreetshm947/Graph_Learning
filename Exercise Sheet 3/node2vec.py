import biased_random_walk as brw
import pickle
import tensorflow as tf

with open('datasets/Citeseer/data.pkl', 'rb') as f:
    graphs_citeseer = pickle.load(f)

G = graphs_citeseer[0]
walks = brw.all_walks(G, 4, 5, 1, 0.1)
anti_walks = brw.negative_sampling(G, walks, 4)
vocab_size = len(G.nodes()) + 1
window_size = 4
embedding_dim = 128
BATCH_SIZE = 1024
BUFFER_SIZE = 2048
AUTOTUNE = tf.data.AUTOTUNE


def convert_list_to_tensor(starting_nodes, node_walks, labels):
    """
    Converts variable length nested list to ragged tensors -> tensors
    :param starting_nodes: List of nodes for each walk
    :param node_walks: Nested list of (context = (positive + negative walk)) for every starting_node
    :param labels: Nested list of labels (0 or 1) for every (starting_node, node_walk) pair.
    E.g. label[i][j] = 1 for (starting_node, walk) and 0 for (starting_node, anti-walk)
    :return: tensor format
    """
    starting_nodes = tf.convert_to_tensor(starting_nodes)

    # Converting from ragged tensor to tensor object by padding with 0
    node_walks = tf.ragged.constant(node_walks).to_tensor(default_value=0,
                                                          shape=[None, len(max(node_walks, key=len))])
    node_walks = tf.reshape(node_walks, (len(node_walks), len(max(node_walks, key=len)), 1))
    labels = tf.ragged.constant(labels).to_tensor(default_value=0,
                                                  shape=[None, len(max(node_walks, key=len))])

    return starting_nodes, node_walks, labels


def create_dataset(starting_nodes, node_walks, labels):
    """
    Creates a tf.data.Dataset object.
    :param starting_nodes: List of nodes for each walk
    :param node_walks: Nested list of (context = (positive + negative walk)) for every starting_node
    :param labels: Nested list of labels (0 or 1) for every (starting_node, node_walk) pair.
    :return: (target_word, context_word), (label)
    """
    dataset = tf.data.Dataset.from_tensor_slices(((starting_nodes, node_walks), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset


def create_training_samples(walks, anti_walks):
    """
    Creates positive and negative skip-gram samples with their resp. output labels
    :param walks: List of all random walks
    :param anti_walks: Negatively sampled walks where len(anti_walks[i] = len(walks[i])
    :return: starting_nodes: (e.g. starting_nodes[:4] = [1,1,1,1,1])
    node_walks: (e.g. node_walks[0] = list(walks[0] + anti_walks[0]))
    labels: (e.g. labels[0] = list(len(walks[0])*1 + len(anti_walks[0])*0)
    """
    starting_nodes, node_walks, labels = [], [], []

    for i in range(len(walks)):

        # Creating positive skip-gram pairs between starting_node and its corresponding random walk
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            walks[i],
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0)

        psg = list(set(tuple(sub) for sub in positive_skip_grams))  # Removing duplicate skip gram pairs
        n, context, label, start = [], [], [], []

        # Creating list skip-grams for only starting_node[j], walks[j]
        for j in range(len(psg)):
            if psg[j][0] == walks[i][0]:
                start, context = psg[j]
                n.append(context)

        # label = 1*positive random walk sample and 0 for negative sample
        label = [1] * len(n)
        label.extend([0] * len(anti_walks[i][:len(n)]))
        n.extend(anti_walks[i][:len(n)])

        starting_nodes.append(start)
        node_walks.append(n)
        labels.append(label)

    return convert_list_to_tensor(starting_nodes, node_walks, labels)  # Converting to tensor objects


starting_nodes, node_walks, labels = create_training_samples(walks, anti_walks)

dataset = create_dataset(starting_nodes, node_walks, labels)


class Node2Vec(tf.keras.models.Model):
    """
    A keras model for learning node embeddings of a graph
    """

    def __init__(self, vocab_size, embedding_dim):
        super(Node2Vec, self).__init__()
        self.node_embedding = tf.keras.layers.Embedding(vocab_size,
                                                        embedding_dim,
                                                        input_length=1)
        self.walk_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dots = tf.keras.layers.Dot(axes=(3, 2))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, pair, **kwargs):
        node, walk = pair
        node_emb = self.node_embedding(node)  # Lookup Embeddings for starting_node
        context_emb = self.walk_embedding(walk)  # Lookup Embeddings for context node
        dots = self.dots([context_emb, node_emb])  # Dot similarity between starting and context embeddings
        return self.flatten(dots)


node2vec = Node2Vec(vocab_size, embedding_dim)

node2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=['accuracy'])

node2vec.fit(dataset, epochs=20)
