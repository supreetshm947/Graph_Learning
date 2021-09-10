from gensim.models import Word2Vec

def learn_embeddings(walks, epochs):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, sg=1, workers=8, vector_size=128, epochs=epochs)

    return model
