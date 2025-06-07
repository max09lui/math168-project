from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def speed_matrix(N):
    assert N in [228, 1026]
    return pd.read_csv(f'STGCN_IJCAI-18-master/dataset/PeMSD7_Full/PeMSD7_V_{N}.csv', header=None).values

def _weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W
    
def weight_matrix(N, weighted=True):
    assert N in [228, 1026]
    file_path = f'STGCN_IJCAI-18-master/dataset/PeMSD7_Full/PeMSD7_W_{N}.csv'
    W = _weight_matrix(file_path)
    if not weighted:
        W[W > 0] = 1
    return W
# A = weight_matrix('STGCN_IJCAI-18-master/dataset/PeMSD7_Full/PeMSD7_W_228.csv')
# plt.imshow(A, cmap="YlGnBu")
# plt.colorbar()
# plt.show()

def add_station_info(G):
    df = pd.read_csv('STGCN_IJCAI-18-master/dataset/PeMSD7_M_Station_Info.csv')
    # remove first column (no need for explicit id column)
    df = df.iloc[:, 1:]
    node_attr = df.to_dict('index')
    nx.set_node_attributes(G, node_attr)

def add_avg_velocity(G, N):
    assert N in [228, 1026]
    V = pd.read_csv(f'STGCN_IJCAI-18-master/dataset/PeMSD7_Full/PeMSD7_V_{N}.csv', header=None).values
    V_avg = V.mean(axis=0)
    V_avg = dict(enumerate(V_avg))
    nx.set_node_attributes(G, V_avg, 'avg_speed')


def networkx_graph(N, weighted=True):
    assert N in [228, 1026]
    A = weight_matrix(N, weighted=weighted)
    G = nx.from_numpy_array(A)
    if N == 228:
        add_station_info(G)
    add_avg_velocity(G, N)
    # print(G.nodes[0])
    return G
