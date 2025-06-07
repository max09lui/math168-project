

import os
from matplotlib import pyplot as plt
import numpy as np
import torch

from net import LinearDynamicalSystem, LogisticDynamicalSystem
from networkx_graph import speed_matrix, weight_matrix

from traffic_animation import plot_map_connection_values, plot_map_connections, speed_matrix
from traffic_animation import animate_map_values, avg_day_speed_at_time, map_values
from centralities import correlation

save_dir = "images"
def plot_weights(X, title, filename):
    X = X.detach().cpu().numpy()
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.clf()

N = 228
A = weight_matrix(N, weighted=True)
A = torch.Tensor(A, device="cpu")
DT = 5 # 5 minutes
    
plot_weights(A, "Adjacency matrix for 228-node graph", "adjacency_matrix_228.png")
A_1026 = torch.Tensor(weight_matrix(1026, weighted=True))
plot_weights(A_1026, "Adjacency matrix for 1026-node graph", "adjacency_matrix_1026.png")

net = LogisticDynamicalSystem(A, timestep=DT)
checkpoint_path = "dynamical_system/logistic_2023-12-03_19-53-35/checkpoint.pth"
# checkpoint_path = "dynamical_system/linear_2023-12-03_19-54-37/checkpoint.pth"
checkpoint_name = "logistic"

checkpoint = torch.load(checkpoint_path)

def plot_loss(train_losses):
    plt.semilogy(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.savefig(os.path.join(save_dir, f"{checkpoint_name}_loss.png"), dpi=300)
    plt.clf()

plot_loss(checkpoint['train_losses'])

net.load_state_dict(checkpoint['model_state_dict'])

for param in net.parameters():
    param.requires_grad = False

bias = net.bias.detach().cpu().numpy()
caps = net.caps.detach().cpu().numpy()
weight = net.weight.detach().cpu().numpy()

map_values(bias, "Network biases", "Dynamical system bias value for all stations", f"{checkpoint_name}_map_net_bias.png", cmap='coolwarm')
map_values(caps, "Network saturating speed", "Dynamical system saturating speed value for all stations", f"{checkpoint_name}_map_net_caps.png", cmap='coolwarm')
total_nondiagonal_weight = weight.sum(axis=0) - weight.diagonal()
map_values(total_nondiagonal_weight, "Network weights", "Dynamical system weight value for all stations", f"{checkpoint_name}_map_net_weight_nodes.png", cmap='coolwarm')
weight_sym = (weight + weight.T) / 2
# zero out the lower triangle
weight_right = weight * (1-np.tri(*weight.shape))
weight_right = weight_right + weight_right.T
# weight_left = weight * np.tri(*weight.shape, -1)
# weight_left = weight_left + weight_left.T
# weight_sym = (np.abs(weight) + np.abs(weight.T)) / 2
plot_map_connection_values(weights=weight_right, title="Map of Network Weights along Edge Connections", filename=f"{checkpoint_name}_map_net_weight_edges.png")

# now make a directed networkx graph with weight as the adjacency matrix
# import networkx as nx
# G = nx.from_numpy_array(weight, create_using=nx.DiGraph)
# G.remove_edges_from([(u, v) for u, v, w in G.edges(data='weight') if w == 0])
# # plot the graph
# plt.title(f"Directed graph of network weights")
# edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
# # Remove self-loops from the graph
# G.remove_edges_from(nx.selfloop_edges(G))
# Plot the modified graph
# nx.draw(G, node_color='red', edge_color=edge_colors, node_size=4)
# plt.savefig(os.path.join(save_dir, f'graph_net_weight_edges.png'), dpi=300)

V = speed_matrix(N)
average_speeds = V.mean(axis=0)
speed_7_50 = avg_day_speed_at_time(7, 50)
speed_5_20 = avg_day_speed_at_time(5, 20)
speed_17_30 = avg_day_speed_at_time(17, 30)
speed_21_50 = avg_day_speed_at_time(21, 50)
correlation(bias, average_speeds, "Network Bias", "Average speed", f"{checkpoint_name}_correlation_speed_net_bias.png")
correlation(bias, speed_7_50, "Network Bias", "Speed at 7:50 AM", f"{checkpoint_name}_correlation_speed_7_50_net_bias.png")
correlation(bias, speed_5_20, "Network Bias", "Speed at 5:20 AM", f"{checkpoint_name}_correlation_speed_5_20_net_bias.png")
correlation(bias, speed_17_30, "Network Bias", "Speed at 5:30 PM", f"{checkpoint_name}_correlation_speed_17_30_net_bias.png")
correlation(bias, speed_21_50, "Network Bias", "Speed at 9:50 PM", f"{checkpoint_name}_correlation_speed_21_50_net_bias.png")
quit()

V = speed_matrix(N)
V = torch.Tensor(V, device="cpu")

def simulate_matrix(net):
    with torch.no_grad():
        # repeatedly feed the output back into the network
        # to get a full day of simulated traffic
        # frame_5_20 = (7*60+50) // 5
        V_0 = V[0]
        V_sim = [V_0]
        for i in range(288-1):
            V_0 = torch.stack((V_0, torch.zeros(N, device="cpu")), dim=0)
            V_0 = net(V_0)
            V_sim.append(V_0[0])
            V_0 = V_0[0]
        V_sim = torch.stack(V_sim)
    return V_sim

V_sim = simulate_matrix(net)
plot_weights(V_sim, "Simulated Velocity starting at 7:50 AM", f"{checkpoint_name}_simulated_7_50.png")
animate_map_values(V_sim, f"{checkpoint_name}_dynamical_system_final.mp4")
