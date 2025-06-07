import os
import time
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from traffic_animation import avg_day_speed_at_time, speed_matrix, map_values
from networkx_graph import networkx_graph
import networkx as nx
from scipy.stats import pearsonr

data_dir = "images"

V = speed_matrix(228)
G = networkx_graph(228)

average_speeds = V.mean(axis=0)

G = networkx_graph(228)

def plot_correlation(x, y, xlabel, ylabel, filename):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # add line of best fit
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', label=f'y = {m:.4f}x + {b:.4f}')
    plt.legend()
    
    plt.savefig(os.path.join(data_dir, filename), dpi=300)
    plt.clf()

def correlation(x, y, xname, yname, filename):
    res = pearsonr(x, y)
    print("Correlation between {} and {}:\n".format(xname, yname), "{: .4f}".format(res[0]), "p-value:", "{: .4f}".format(res[1]))
    plot_correlation(x, y, xname, yname, filename)

degree_centrality = nx.degree_centrality(G)
degree_centrality = np.array([degree_centrality[n] for n in G.nodes()])

eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
eigenvector_centrality = np.array([eigenvector_centrality[n] for n in G.nodes()])

katz_centrality = nx.katz_centrality_numpy(G)
katz_centrality = np.array([katz_centrality[n] for n in G.nodes()])

page_rank_centrality = nx.pagerank(G)
page_rank_centrality = np.array([page_rank_centrality[n] for n in G.nodes()])

# Plot vertical lines
def plot_vertical_line(hours, minutes, color):
    plt.axvline(x=hours*60+minutes, color=color, linestyle='--')
    plt.text(hours*60+minutes, plt.ylim()[1], f'{hours:02d}:{minutes:02d}', color=color, ha='center', va='bottom')

def plot_quantity_over_24_hours(quantity, ylabel, title, filename, vertical_lines=None):
    X = np.arange(0, 288 * 5, 5)
    plt.plot(X, quantity)
    plt.xlabel("Time of day (minutes)")
    plt.ylabel(ylabel)
    plt.title(title, pad=15)
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M', time.gmtime(ms * 60)))
    plt.xticks(np.arange(0, 60*25, 3*60))
    plt.gca().xaxis.set_major_formatter(formatter)

    if vertical_lines:
        for line in vertical_lines:
            plot_vertical_line(line[0], line[1], line[2])

    plt.savefig(os.path.join(data_dir, filename), dpi=300)
    plt.clf()

vertical_lines = [
    (7, 50, 'red'),
    (17, 30, 'red'),
    (5, 20, 'green'),
    (13, 10, 'green'),
    (21, 50, 'green')
]

def correlations_by_time_of_day(centrality, centrality_name, filename):
    correlations = []
    p_values = []
    for t in range(288):
        speeds_at_t = avg_day_speed_at_time(t // 12, (t % 12) * 5)
        res = pearsonr(speeds_at_t, centrality)
        correlations.append(res[0])
        p_values.append(res[1])

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    plot_quantity_over_24_hours(correlations, "Correlation", "Correlation between speed and {} vs time of day".format(centrality_name), filename, vertical_lines=vertical_lines)
    plot_quantity_over_24_hours(p_values, "P-value", "P-values for correlation between speed and {}".format(centrality_name), filename.replace(".png", "_p_values.png"), vertical_lines=vertical_lines)


if __name__ == "__main__":
    correlation(average_speeds, degree_centrality, "Average speed", "Degree", "correlation_speed_degree.png")
    correlation(average_speeds, eigenvector_centrality, "Average speed", "Eigenvector centrality", "correlation_speed_eigenvector_centrality.png")
    correlation(average_speeds, katz_centrality, "Average speed", "Katz centrality", "correlation_speed_katz_centrality.png")
    correlation(average_speeds, page_rank_centrality, "Average speed", "Page rank centrality", "correlation_speed_page_rank_centrality.png")

    speeds_at_7_50 = avg_day_speed_at_time(7, 50)
    correlation(speeds_at_7_50, degree_centrality, "Speed at 7:50", "Degree", "correlation_speed_7_50_degree.png")
    correlation(speeds_at_7_50, page_rank_centrality, "Speed at 7:50", "Page rank centrality", "correlation_speed_7_50_page_rank_centrality.png")
    speed_at_17_30 = avg_day_speed_at_time(17, 30)
    correlation(speed_at_17_30, degree_centrality, "Speed at 17:30", "Degree", "correlation_speed_17_30_degree.png")
    correlation(speed_at_17_30, page_rank_centrality, "Speed at 17:30", "Page rank centrality", "correlation_speed_17_30_page_rank_centrality.png")
    speed_at_13_10 = avg_day_speed_at_time(13, 10)
    correlation(speed_at_13_10, degree_centrality, "Speed at 13:10", "Degree", "correlation_speed_13_10_degree.png")
    correlation(speed_at_13_10, page_rank_centrality, "Speed at 13:10", "Page rank centrality", "correlation_speed_13_10_page_rank_centrality.png")

    map_values(eigenvector_centrality, "Eigenvector centrality", "Eigenvector centrality of all stations", "map_eigenvector_centrality.png")
    map_values(katz_centrality, "Katz centrality", "Katz centrality of all stations", "map_katz_centrality.png")
    map_values(degree_centrality, "Degree centrality", "Degree centrality of all stations", "map_degree_centrality.png")
    map_values(page_rank_centrality, "Page rank centrality", "Page rank centrality of all stations", "map_page_rank_centrality.png")

    correlations_by_time_of_day(degree_centrality, "degree centrality", "correlation_by_time_of_day_speed_degree_centrality.png")
    correlations_by_time_of_day(page_rank_centrality, "page rank centrality", "correlation_by_time_of_day_speed_page_rank_centrality.png")
    correlations_by_time_of_day(eigenvector_centrality, "eigenvector centrality", "correlation_by_time_of_day_speed_eigenvector_centrality.png")
    correlations_by_time_of_day(katz_centrality, "katz centrality", "correlation_by_time_of_day_speed_katz_centrality.png")