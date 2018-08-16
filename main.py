# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group

from sklearn.metrics import roc_auc_score
import math
import subprocess
import Node2Vec_LayerSelect

import argparse
from MNE import *

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', type=int, default=10,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


# randomly divide data into few parts for the purpose of cross-validation
def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]


def randomly_choose_false_edges(nodes, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
    return tmp_list


def get_dict_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()


def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


def get_common_neighbor_score(networks, target_A, target_B):
    common_neighbor_counter = 0
    tmp_network = networks
    A_neighbors = list()
    B_neighbors = list()
    for edge in tmp_network:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    return common_neighbor_counter


def get_Jaccard_score(networks, target_A, target_B):
    tmp_network = networks
    A_neighbors = list()
    B_neighbors = list()
    for edge in tmp_network:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    common_neighbor_counter = 0
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    if len(A_neighbors) == 0 and len(B_neighbors) == 0:
        Jaccard_score = 1
    else:
        Jaccard_score = common_neighbor_counter/(len(A_neighbors) + len(B_neighbors) - common_neighbor_counter)
    return Jaccard_score


def get_frequency_dict(networks):
    counting_dict = dict()
    for edge in networks:
        if edge[0] not in counting_dict:
            counting_dict[edge[0]] = 0
        if edge[1] not in counting_dict:
            counting_dict[edge[1]] = 0
        counting_dict[edge[0]] += 1
        counting_dict[edge[1]] += 1
    return counting_dict


def get_AA_score(networks, target_A, target_B, frequency_dict):
    AA_score = 0
    A_neighbors = list()
    B_neighbors = list()
    for edge in networks:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            if frequency_dict[neighbor] > 1:
                AA_score += 1/(math.log(frequency_dict[neighbor]))
    return AA_score


def read_LINE_vectors(file_name):
    tmp_embedding = dict()
    file = open(file_name, 'r')
    for line in file.readlines()[1:]:
        numbers = line[:-2].split(' ')
        tmp_vector = list()
        for n in numbers[1:]:
            tmp_vector.append(float(n))
            tmp_embedding[numbers[0]] = np.asarray(tmp_vector)
    file.close()
    return tmp_embedding


def train_LINE_model(edges, epoch_num=1, dimension=100, negative=5):
    preparation_command = 'LD_LIBRARY_PATH=/usr/local/lib\nexport LD_LIBRARY_PATH'
    file_name = 'LINE_tmp_edges.txt'
    file = open(file_name, 'w')
    for edge in edges:
        file.write(edge[0] + ' ' + edge[1] + ' 1\n')
    file.close()
    command1 = 'C++/LINE/linux/line -train LINE_tmp_edges.txt -output LINE_tmp_embedding1.txt -order 1 base-negative ' + str(
        negative) + ' -dimension ' + str(dimension / 2)
    command2 = 'C++/LINE/linux/line -train LINE_tmp_edges.txt -output LINE_tmp_embedding2.txt -order 2 -negative ' + str(
        negative) + ' -dimension ' + str(dimension / 2)
    subprocess.call(preparation_command + '\n' + command1 + '\n' + command2, shell=True)
    print('finish training')
    first_order_embedding = read_LINE_vectors('LINE_tmp_embedding1.txt')
    second_order_embedding = read_LINE_vectors('LINE_tmp_embedding2.txt')
    final_embedding = dict()
    for node in first_order_embedding:
        final_embedding[node] = np.append(first_order_embedding[node], second_order_embedding[node])
    return final_embedding


def Evaluate_basic_methods(input_network):
    print('Start to analyze the base methods')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = list()
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.append(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.append(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    print('We are analyzing the common neighbor method')
    all_network = set(all_network)
    true_list = list()
    prediction_list = list()
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_common_neighbor_score(all_network, edge[0], edge[1]))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_common_neighbor_score(all_network, edge[0], edge[1]))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    common_neighbor_performance = roc_auc_score(y_true, y_scores)
    print('Performance of common neighbor:', common_neighbor_performance)
    print('We are analyzing the Jaccard method')
    true_list = list()
    prediction_list = list()
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_Jaccard_score(all_network, edge[0], edge[1]))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_Jaccard_score(all_network, edge[0], edge[1]))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    Jaccard_performance = roc_auc_score(y_true, y_scores)
    print('Performance of Jaccard:', Jaccard_performance)
    print('We are analyzing the AA method')
    true_list = list()
    prediction_list = list()
    frequency_dict = get_frequency_dict(all_network)
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_AA_score(all_network, edge[0], edge[1], frequency_dict))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_AA_score(all_network, edge[0], edge[1], frequency_dict))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    AA_performance = roc_auc_score(y_true, y_scores)
    print('Performance of AA:', AA_performance)
    return common_neighbor_performance, Jaccard_performance, AA_performance


def merge_PMNE_models(input_all_models, all_nodes):
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate((final_model[node], tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), axis=0)
                else:
                    final_model[node] = np.concatenate((final_model[node], np.zeros([args.dimensions])), axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([args.dimensions])
    return final_model


def Evaluate_PMNE_methods(input_network):
    # we need to write codes to implement the co-analysis method of PMNE
    print('Start to analyze the PMNE method')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = list()
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.append(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.append(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    # print('We are working on method one')
    all_network = set(all_network)
    G = Random_walk.RWGraph(get_G_from_edges(all_network), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_one = train_deepwalk_embedding(walks)
    method_one_performance = get_AUC(model_one, all_test_network, all_false_network)
    print('Performance of PMNE method one:', method_one_performance)
    # print('We are working on method two')
    all_models = list()
    for edge_type in training_network:
        tmp_edges = training_network[edge_type]
        tmp_G = Random_walk.RWGraph(get_G_from_edges(tmp_edges), args.directed, args.p, args.q)
        tmp_G.preprocess_transition_probs()
        walks = tmp_G.simulate_walks(args.num_walks, args.walk_length)
        tmp_model = train_deepwalk_embedding(walks)
        all_models.append(tmp_model)
    model_two = merge_PMNE_models(all_models, all_nodes)
    method_two_performance = get_dict_AUC(model_two, all_test_network, all_false_network)
    print('Performance of PMNE method two:', method_two_performance)
    # print('We are working on method three')
    tmp_graphs = list()
    for edge_type in training_network:
        tmp_G = get_G_from_edges(training_network[edge_type])
        tmp_graphs.append(tmp_G)
    MK_G = Node2Vec_LayerSelect.Graph(tmp_graphs, args.p, args.q, 0.5)
    MK_G.preprocess_transition_probs()
    MK_walks = MK_G.simulate_walks(args.num_walks, args.walk_length)
    model_three = train_deepwalk_embedding(MK_walks)
    method_three_performance = get_AUC(model_three, all_test_network, all_false_network)
    print('Performance of PMNE method three:', method_three_performance)
    return method_one_performance, method_two_performance, method_three_performance

args = parse_args()
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_name = sys.argv[1]
# file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, _, all_nodes = load_network_data(file_name)
# model = train_model(edge_data_by_type)

# In our experiment, we use 5-fold cross-validation, but you can change that
number_of_groups = 5
edge_data_by_type_by_group = dict()
for edge_type in edge_data_by_type:
    all_data = edge_data_by_type[edge_type]
    separated_data = divide_data(all_data, number_of_groups)
    edge_data_by_type_by_group[edge_type] = separated_data

overall_MNE_performance = list()
overall_node2Vec_performance = list()
overall_LINE_performance = list()
overall_Deepwalk_performance = list()
overall_common_neighbor_performance = list()
overall_Jaccard_performance = list()
overall_AA_performance = list()
overall_PMNE_1_performance = list()
overall_PMNE_2_performance = list()
overall_PMNE_3_performance = list()

for i in range(number_of_groups):
    training_data_by_type = dict()
    evaluation_data_by_type = dict()
    for edge_type in edge_data_by_type_by_group:
        training_data_by_type[edge_type] = list()
        evaluation_data_by_type[edge_type] = list()
        for j in range(number_of_groups):
            if j == i:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    evaluation_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
            else:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    training_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
    base_edges = list()
    training_nodes = list()
    for edge_type in training_data_by_type:
        for edge in training_data_by_type[edge_type]:
            base_edges.append(edge)
            training_nodes.append(edge[0])
            training_nodes.append(edge[1])
    training_nodes = list(set(training_nodes))
    training_data_by_type['Base'] = base_edges
    MNE_model = train_model(training_data_by_type)

    tmp_MNE_performance = 0
    tmp_node2Vec_performance = 0
    tmp_LINE_performance = 0
    tmp_Deepwalk_performance = 0
    merged_networks = dict()
    merged_networks['training'] = dict()
    merged_networks['test_true'] = dict()
    merged_networks['test_false'] = dict()
    for edge_type in training_data_by_type:
        if edge_type == 'Base':
            continue
        print('We are working on edge:', edge_type)
        selected_true_edges = list()
        tmp_training_nodes = list()
        for edge in training_data_by_type[edge_type]:
            tmp_training_nodes.append(edge[0])
            tmp_training_nodes.append(edge[1])
        tmp_training_nodes = set(tmp_training_nodes)
        for edge in evaluation_data_by_type[edge_type]:
            if edge[0] in tmp_training_nodes and edge[1] in tmp_training_nodes:
                if edge[0] == edge[1]:
                    continue
                selected_true_edges.append(edge)
        if len(selected_true_edges) == 0:
            continue
        selected_false_edges = randomly_choose_false_edges(training_nodes, edge_data_by_type[edge_type])
        print('number of info network edges:', len(training_data_by_type[edge_type]))
        print('number of evaluation edges:', len(selected_true_edges))
        merged_networks['training'][edge_type] = set(training_data_by_type[edge_type])
        merged_networks['test_true'][edge_type] = selected_true_edges
        merged_networks['test_false'][edge_type] = selected_false_edges

        local_model = dict()
        for pos in range(len(MNE_model['index2word'])):
            # 0.5 is the weight parameter mentioned in the paper, which is used to show how important each relation type is and can be tuned based on the network.
            local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.5*np.dot(MNE_model['addition'][edge_type][pos], MNE_model['tran'][edge_type])
        tmp_MNE_score = get_dict_AUC(local_model, selected_true_edges, selected_false_edges)
        # tmp_MNE_score = get_AUC(MNE_model['addition'][edge_type], selected_true_edges, selected_false_edges)
        print('MNE score:', tmp_MNE_score)
        node2vec_G = Random_walk.RWGraph(get_G_from_edges(training_data_by_type[edge_type]), args.directed, 2, 0.5)
        node2vec_G.preprocess_transition_probs()
        node2vec_walks = node2vec_G.simulate_walks(20, 10)
        node2vec_model = train_deepwalk_embedding(node2vec_walks)
        tmp_node2vec_score = get_AUC(node2vec_model, selected_true_edges, selected_false_edges)
        print('node2vec score:', tmp_node2vec_score)
        Deepwalk_G = Random_walk.RWGraph(get_G_from_edges(training_data_by_type[edge_type]), args.directed, 1, 1)
        Deepwalk_G.preprocess_transition_probs()
        Deepwalk_walks = Deepwalk_G.simulate_walks(args.num_walks, 10)
        Deepwalk_model = train_deepwalk_embedding(Deepwalk_walks)
        tmp_Deepwalk_score = get_AUC(Deepwalk_model, selected_true_edges, selected_false_edges)
        print('Deepwalk score:', tmp_Deepwalk_score)
        LINE_model = train_LINE_model(training_data_by_type[edge_type])
        tmp_LINE_score = get_dict_AUC(LINE_model, selected_true_edges, selected_false_edges)
        print('LINE score:', tmp_LINE_score)
        tmp_MNE_performance += tmp_MNE_score
        tmp_node2Vec_performance += tmp_node2vec_score
        tmp_LINE_performance += tmp_LINE_score
        tmp_Deepwalk_performance += tmp_Deepwalk_score

    print('MNE performance:', tmp_MNE_performance / (len(training_data_by_type)-1))
    print('node2vec performance:', tmp_node2Vec_performance / (len(training_data_by_type)-1))
    print('LINE performance:', tmp_LINE_performance / (len(training_data_by_type)-1))
    print('Deepwalk performance:', tmp_Deepwalk_performance / (len(training_data_by_type)-1))
    overall_MNE_performance.append(tmp_MNE_performance / (len(training_data_by_type)-1))
    overall_node2Vec_performance.append(tmp_node2Vec_performance / (len(training_data_by_type)-1))
    overall_LINE_performance.append(tmp_LINE_performance / (len(training_data_by_type)-1))
    overall_Deepwalk_performance.append(tmp_Deepwalk_performance / (len(training_data_by_type)-1))
    common_neighbor_performance, Jaccard_performance, AA_performance = Evaluate_basic_methods(merged_networks)
    performance_1, performance_2, performance_3 = Evaluate_PMNE_methods(merged_networks)
    overall_common_neighbor_performance.append(common_neighbor_performance)
    overall_Jaccard_performance.append(Jaccard_performance)
    overall_AA_performance.append(AA_performance)
    overall_PMNE_1_performance.append(performance_1)
    overall_PMNE_2_performance.append(performance_2)
    overall_PMNE_3_performance.append(performance_3)

overall_MNE_performance = np.asarray(overall_MNE_performance)
overall_node2Vec_performance = np.asarray(overall_node2Vec_performance)
overall_LINE_performance = np.asarray(overall_LINE_performance)
overall_Deepwalk_performance = np.asarray(overall_Deepwalk_performance)
overall_common_neighbor_performance = np.asarray(overall_common_neighbor_performance)
overall_Jaccard_performance = np.asarray(overall_Jaccard_performance)
overall_AA_performance = np.asarray(overall_AA_performance)
overall_PMNE_1_performance = np.asarray(overall_PMNE_1_performance)
overall_PMNE_2_performance = np.asarray(overall_PMNE_2_performance)
overall_PMNE_3_performance = np.asarray(overall_PMNE_3_performance)

print('Overall MRNE AUC:', overall_MNE_performance)
print('Overall node2Vec AUC:', overall_node2Vec_performance)
print('Overall LINE AUC:', overall_LINE_performance)
print('Overall Deepwalk AUC:', overall_Deepwalk_performance)
print('Overall Common neighbor AUC:', overall_common_neighbor_performance)
print('Overall Jaccard AUC:', overall_Jaccard_performance)
print('Overall AA AUC:', overall_AA_performance)
print('Overall PMNE 1 AUC:', overall_PMNE_1_performance)
print('Overall PMNE 2 AUC:', overall_PMNE_2_performance)
print('Overall PMNE 3 AUC:', overall_PMNE_3_performance)

print('')
print('')
print('')

print('Overall MRNE AUC:', np.mean(overall_MNE_performance))
print('Overall node2Vec AUC:', np.mean(overall_node2Vec_performance))
print('Overall LINE AUC:', np.mean(overall_LINE_performance))
print('Overall Deepwalk AUC:', np.mean(overall_Deepwalk_performance))
print('Overall Common neighbor AUC:', np.mean(overall_common_neighbor_performance))
print('Overall Jaccard AUC:', np.mean(overall_Jaccard_performance))
print('Overall AA AUC:', np.mean(overall_AA_performance))
print('Overall PMNE 1 AUC:', np.mean(overall_PMNE_1_performance))
print('Overall PMNE 2 AUC:', np.mean(overall_PMNE_2_performance))
print('Overall PMNE 3 AUC:', np.mean(overall_PMNE_3_performance))

print('')
print('')
print('')

print('Overall MRNE std:', np.std(overall_MNE_performance))
print('Overall node2Vec std:', np.std(overall_node2Vec_performance))
print('Overall LINE std:', np.std(overall_LINE_performance))
print('Overall Deepwalk std:', np.std(overall_Deepwalk_performance))
print('Overall Common neighbor std:', np.std(overall_common_neighbor_performance))
print('Overall Jaccard std:', np.std(overall_Jaccard_performance))
print('Overall AA std:', np.std(overall_AA_performance))
print('Overall PMNE 1 std:', np.std(overall_PMNE_1_performance))
print('Overall PMNE 2 std:', np.std(overall_PMNE_2_performance))
print('Overall PMNE 3 std:', np.std(overall_PMNE_3_performance))

print('end')
