import GMAN_node2vec as node2vec
import pandas as pd 
import numpy as np 
import networkx as nx
from gensim.models import Word2Vec
# pip install gensim

# SENSOR_IDS_PATH='../PEMSBAY/pemsbay_ids.txt'
# DISTANCE_PATH='../PEMSBAY/distances_bay_2017.csv'
# ADJPATH = '../PEMSBAY/adj_pemsbay_tmp.txt'

# def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
#     """
#     :param distance_df: data frame with three columns: [from, to, distance].
#     :param sensor_ids: list of sensor ids.
#     :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
#     :return: 
#     """

#     num_sensors = len(sensor_ids)
#     dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
#     dist_mx[:] = np.inf
#     # Builds sensor id to index map.
#     sensor_id_to_ind = {}
#     for i, sensor_id in enumerate(sensor_ids):
#         sensor_id_to_ind[sensor_id] = i
    
#     # Fills cells in the matrix with distances.
#     for row in distance_df.values:
#         if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
#             continue
#         dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

#     # Calculates the standard deviation as theta.
#     distances = dist_mx[~np.isinf(dist_mx)].flatten()
#     std = distances.std()
#     adj_mx = np.exp(-np.square(dist_mx / std))

#     # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
#     adj_mx[adj_mx < normalized_k] = 0
#     return adj_mx

# def gen_matrix(sensor_ids_filename=SENSOR_IDS_PATH, 
#     distances_filename=DISTANCE_PATH, 
#     normalized_k=0.1):
#     with open(sensor_ids_filename) as f:
#         sensor_ids = f.read().strip().split(',')
#     distance_df = pd.read_csv(distances_filename, dtype={'from': 'str', 'to': 'str'})
#     adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
#     return adj_mx

def get_adj(adj_path):
    adj = np.load(adj_path)
    adj_pad = np.zeros((adj.shape[0] + 1, adj.shape[1] + 1))
    adj_pad[:adj.shape[0], :adj.shape[1]] = adj
    np.fill_diagonal(adj_pad, 1)
    return adj_pad

def gen_edgelist(matrix):
    assert matrix.shape[0] == matrix.shape[1], "matrix should be N*N size"
    N_NODE = matrix.shape[0]
    edge_list = []
    for i in range(N_NODE):
        for j in range(N_NODE):
            row = [int(i), int(j), matrix[i][j]]
            edge_list.append(row)
    np.savetxt(EDGE_PATH, edge_list, fmt='%d %d %f')

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size = dimensions, window = 10, min_count=0, sg=1, workers = 8, epochs = ITER)
    model.wv.save_word2vec_format(output_file)
    return
    
# parameters(generate spatial embedding).
ADJ_PATH = '../data/adjacency_matrix.npy'
SE_PATH = '../data/Japan_prefectures_embedding.txt'
EDGE_PATH = '../data/Japan_prefectures_edgelist.txt'
SE_DIM = 64
IS_DIRECTED = True
P = 2
Q = 1
NUM_WALKS = 100
WALK_LENGTH = 20
WINDOW_SIZE = 10
ITER = 1000

def main():
    adj_mx = get_adj(ADJ_PATH)
    gen_edgelist(adj_mx)
    nx_G = read_graph(EDGE_PATH)
    G = node2vec.Graph(nx_G, IS_DIRECTED, P, Q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(NUM_WALKS, WALK_LENGTH)
    learn_embeddings(walks, SE_DIM, SE_PATH)


if __name__ == '__main__':
    main()


