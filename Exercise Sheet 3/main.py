import biased_random_walk as brw
import pickle
import node_classification as nc
import link_prediction as lp

#  Node Classification for CORA graph dataset

with open('datasets/Cora/data.pkl', 'rb') as f:
    graphs_cora = pickle.load(f)

G_cora = graphs_cora[0]
print("Node Classification Cora Dataset")

nc.classify(G_cora, 4, 5, 1, 1)
nc.classify(G_cora, 4, 5, .1, 1)
nc.classify(G_cora, 4, 5, 1, .1)

#  Node Classification for CITESEER graph dataset

with open('datasets/Citeseer/data.pkl', 'rb') as f:
    graphs_citeseer = pickle.load(f)

G_citeseer = graphs_citeseer[0]
print("Node Classification Citiseer Dataset")

nc.classify(G_citeseer, 4, 5, 1, 1)
nc.classify(G_citeseer, 4, 5, .1, 1)
nc.classify(G_citeseer, 4, 5, 1, .1)

# Link Prediction

#  PPI graph dataset
with open('datasets/PPI/data.pkl', 'rb') as f:
    graphs_ppi = pickle.load(f)

print("Link Prediction PPI Dataset")

G_ppi = graphs_ppi[0]
lp.link_prediction(G_ppi)

#  FACEBOOK graph dataset
with open('datasets/Facebook/data.pkl', 'rb') as f:
    graphs_fb = pickle.load(f)

print("Link Prediction FB Dataset")

G_fb = graphs_fb[0]
lp.link_prediction(G_fb)