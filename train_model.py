# This python file is used to train the Multiplex Network Embedding model
# Author: Hongming ZHANG, HKUST KnowComp Group

import networkx as nx
import Random_walk
from MNE import *
import sys


file_name = sys.argv[1]
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
model = train_model(edge_data_by_type)
# print(model)
save_model(model, 'model')
save_embeddings(model, 'model')

test_model = load_model('model')
# print(test_model)

# print('end')
