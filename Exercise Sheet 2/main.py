import node_classification
import pickle
import data_utils
import os
from tensorflow import keras

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

print("Node Classification ---- CITESEER DATASET ---- ")

with open('datasets/Citeseer_Train/data.pkl', 'rb') as f:
    graphs_citeseer = pickle.load(f)

with open('datasets/Citeseer_Eval/data.pkl', 'rb') as k:
    graphs_eval_citeseer = pickle.load(k)

# Training GCN model for CITESEER
model_citeseer = node_classification.createModel(graphs_citeseer[0], 600)

# Visualising model architecture
keras.ut.plot_model(model_citeseer, "./Model architecture/2-Layer-GCN-Citeseer.png", show_shapes=True)

# Loading Test Data
X_test_citeseer = data_utils.get_node_attributes(graphs_eval_citeseer[0])
Y_test_citeseer = data_utils.get_node_labels(graphs_eval_citeseer[0]) - 1  # Making the labels start from 0
print(X_test_citeseer.shape, Y_test_citeseer.shape)

# Evaluating GCN model
node_classification.test_model(model_citeseer, X_test_citeseer, Y_test_citeseer)
print("-------------------------------------------------------------------------")

print("Node Classification ---- CORA DATASET ---- ")

with open('datasets/Cora_Train/data.pkl', 'rb') as f:
    graphs_cora = pickle.load(f)

with open('datasets/Cora_Eval/data.pkl', 'rb') as k:
    graphs_eval_cora = pickle.load(k)

# Training GCN model for CORA
model_cora = node_classification.createModel(graphs_cora[0], 1000)

# Visualising model architecture
keras.ut.plot_model(model_cora, "./Model architecture/2-Layer-GCN-Cora.png", show_shapes=True)

# Loading test for CORA
X_test_cora = data_utils.get_node_attributes(graphs_eval_cora[0])
Y_test_cora = data_utils.get_node_labels(graphs_eval_cora[0]) - 1  # Making the labels start from 0

# Evaluating GCN model
node_classification.test_model(model_cora, X_test_cora, Y_test_cora)

print("-------------------------------------------------------------------------")