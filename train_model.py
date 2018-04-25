# This python file is used to train the Multiplex Network Embedding model
# Author: Hongming ZHANG, HKUST KnowComp Group

import networkx as nx
import Random_walk
from MNE import *
import sys


file_name = sys.argv[1]
# test_file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
model = train_model(edge_data_by_type)
save_model(model, 'model')
