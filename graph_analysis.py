import collections
import numpy as np
import matplotlib.pyplot as plt
import os

from networkx_graph import networkx_graph
import networkx as nx

data_dir = "images"

def draw_graph(N):
    assert N in [228, 1026]
    G = networkx_graph(N)
    plt.title(f"Topology of network for {N}-node Graph")
    nx.draw(G, node_color='red', edge_color='blue', node_size=4)
    # nx.draw_kamada_kawai(G, node_color='red', edge_color='blue', node_size=4)
    plt.savefig(os.path.join(data_dir, f'graph_{N}.png'), dpi=300)
    plt.clf()

def plot_degree_distribution(N):
    assert N in [228, 1026]
    G = networkx_graph(N)
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, density=True)
    plt.title(f"Degree Distribution for {N}-node Graph")
    plt.xlabel("Degree k")
    plt.ylabel("Frequency P(k)")
    plt.savefig(os.path.join(data_dir, f"degree_distribution_{N}.png"), dpi=300)
    plt.clf()

def plot_cumulative_degree_distribution(N):
    assert N in [228, 1026]
    G = networkx_graph(N)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)  # Calculate cumulative sum
    plt.loglog(deg, cs, 'bo')
    plt.title(f"Cumulative Degree Distribution for {N}-node Graph")
    plt.xlabel("Degree k")
    plt.ylabel("Cumulative Frequency P(k)")  # Update ylabel
    plt.savefig(os.path.join(data_dir, f"cumulative_degree_distribution_{N}.png"), dpi=300)
    plt.clf()

# draw_graph(228)
# draw_graph(1026)
# plot_degree_distribution(228)
# plot_degree_distribution(1026)
# plot_cumulative_degree_distribution(228)
# plot_cumulative_degree_distribution(1026)

G = networkx_graph(1026)
print("Conn comps", nx.number_connected_components(G))