import datetime
import time
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from networkx_graph import networkx_graph, speed_matrix

import os
from matplotlib.collections import LineCollection
data_dir = "images"

info = pd.read_csv('STGCN_IJCAI-18-master/dataset/PeMSD7_M_Station_Info.csv')
# print(info.head())

longitude_min, longitude_max = info['Longitude'].min(), info['Longitude'].max()
latitude_min, latitude_max = info['Latitude'].min(), info['Latitude'].max()
# add % buffer to each side
longitude_range = longitude_max - longitude_min
latitude_range = latitude_max - latitude_min
longitude_buffer_percent = 0.05
latitude_buffer_percent = 0.05
longitude_buffer = longitude_range * longitude_buffer_percent
latitude_buffer = latitude_range * latitude_buffer_percent
longitude_min -= longitude_buffer
longitude_max += longitude_buffer
latitude_min -= latitude_buffer
latitude_max += latitude_buffer
# floor/ceil to 2 decimal places
longitude_min = np.floor(longitude_min * 100) / 100
longitude_max = np.ceil(longitude_max * 100) / 100
latitude_min = np.floor(latitude_min * 100) / 100
latitude_max = np.ceil(latitude_max * 100) / 100
print(longitude_min, longitude_max)
print(latitude_min, latitude_max)

V = speed_matrix(228)
print("Days in dataset:", V.shape[0] // 288)

def avg_day_speed_matrix(N):
    V = speed_matrix(228)
    V = V.reshape(-1, 288, 228).mean(axis=0)
    return V
avg_day_speeds = avg_day_speed_matrix(228)

def avg_day_speed_at_time(hours, minutes):
    frame_number = (hours * 60 + minutes) // 5
    return avg_day_speeds[frame_number]

def setup_map_plot():
    # plot the map
    img = mpimg.imread(f'map_{longitude_buffer_percent}_{latitude_buffer_percent}.png')
    # avg color channels
    img = img[:,:,:3].mean(axis=2)
    # change x y limits to match the map
    plt.xlim(longitude_min, longitude_max)
    plt.ylim(latitude_min, latitude_max)
    # plot the map in the limits
    plt.imshow(img, extent=[longitude_min, longitude_max, latitude_min, latitude_max], cmap='gray')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

def plot_map_connections():
    setup_map_plot()

    # plot all of the connections
    G = networkx_graph(228)
    for edge in G.edges():
        i, j = edge
        weight = G[i][j]['weight']
        plt.plot([info['Longitude'][i], info['Longitude'][j]], [info['Latitude'][i], info['Latitude'][j]], c='blue', alpha=weight, linewidth=0.8, zorder=0)

    # plot all of the stations
    plt.scatter(info['Longitude'], info['Latitude'], s=5, c='red', edgecolor="black", linewidths=0.5, zorder=1)

    plt.title(f"Map of 228-node Graph with Edge Connections")

    plt.savefig(os.path.join(data_dir, 'map_connections.png'), dpi=300)
    plt.clf()


def plot_map_connection_values(weights, title, filename):
    setup_map_plot()

    # plot all of the connections
    G = networkx_graph(228)
    edges = G.edges()
    edge_weights = [G[i][j]['weight'] if weights is None else weights[i, j] for i, j in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_colors = plt.cm.viridis((np.array(edge_weights) - min_weight) / (max_weight - min_weight))  # Normalize to [0, 1]
    edge_segments = [(np.array([info['Longitude'][i], info['Latitude'][i]]), np.array([info['Longitude'][j], info['Latitude'][j]])) for i, j in edges]
    lc = LineCollection(edge_segments, colors=edge_colors, alpha=1, linewidths=1, zorder=1)
    plt.gca().add_collection(lc)

    # plot all of the stations
    plt.scatter(info['Longitude'], info['Latitude'], s=5, c='red', edgecolor="black", linewidths=0.5, zorder=0)

    # add a colorbar for the weights
    cbar = plt.colorbar(lc, fraction=0.025)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("Weight (normalized to [0,1])", rotation=270)

    plt.title(title)

    plt.savefig(os.path.join(data_dir, filename), dpi=300, bbox_inches="tight")
    plt.clf()

def map_values(c, label, title, filename, cmap=None):
    setup_map_plot()
    # plot all of the stations
    c_abs_max = max(abs(c))
    plt.scatter(info['Longitude'], info['Latitude'], s=50, c=c, edgecolor='black', cmap=cmap, vmin=-c_abs_max, vmax=c_abs_max)

    cbar = plt.colorbar(fraction=0.025)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(label, rotation=270)
    plt.title(title)

    plt.savefig(os.path.join(data_dir, filename), dpi=300, bbox_inches="tight")
    plt.clf()

def map_speeds(speeds, label, title, filename):
    setup_map_plot()
    # plot all of the stations
    plt.scatter(info['Longitude'], info['Latitude'], s=50, c=speeds, cmap='RdYlGn', edgecolor='black', vmin=0, vmax=V.max())

    cbar = plt.colorbar(fraction=0.025)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(label,rotation=270)
    plt.title(title)

    plt.savefig(os.path.join(data_dir, filename), dpi=300, bbox_inches="tight")
    plt.clf()

def map_average_speed():
    map_speeds(V.mean(axis=0), "Average Speed (mph)", "Average Speed of All Stations", "map_average_speed.png")

def map_speeds_at_time(hours, minutes):
    frame_number = (hours * 60 + minutes) // 5
    map_speeds(avg_day_speeds[frame_number], "Speed (mph)", f"Speed of all stations at {hours:02d}:{minutes:02d}", f"map_speeds_{hours:02d}_{minutes:02d}.png")

def frame_to_time(frame_number):
    datapoint = frame_number
    weekdays = datapoint // 288
    minutes = (datapoint % 288) * 5
    weekends = weekdays // 5
    days = weekdays + weekends * 2
    dt = datetime.datetime(2012, 5, 1) + datetime.timedelta(days=days, minutes=minutes)
    return dt

def animate_map_values(V, filename):
    fig = plt.figure()
    setup_map_plot()
    # plot all of the stations
    scat = plt.scatter(info['Longitude'], info['Latitude'], s=50, c=V[0], cmap='RdYlGn', edgecolor='black', vmin=0, vmax=V.max())

    cbar = plt.colorbar(fraction=0.025)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("Speed at station (mph)",rotation=270)

    seconds_per_day = 12
    # each frame is 5 minutes
    days_per_frame = 5 / 60 / 24
    fps = 1 / (seconds_per_day * days_per_frame)
    frame_count = len(V)
    scale = 3

    def update(frame_number):
        # scat.set_sizes(V[frame_number] * scale)
        scat.set_array(V[frame_number])
        plt.title(frame_to_time(frame_number).strftime("%A %B %d, %Y %I:%M %p"))

    ani = FuncAnimation(fig, update, interval=1000/fps, save_count=frame_count)
    ani.save(os.path.join(data_dir, filename))
    plt.close()
    plt.clf()

def plot_speed_24_hours():
    # time in minutes
    X = np.arange(0, 288 * 5, 5)
    plt.plot(X, avg_day_speeds.mean(axis=1))
    plt.xlabel("Time of day (minutes)")
    plt.ylabel("Average speed (mph)")
    plt.title("Average speed vs time of day", pad=15) 
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M', time.gmtime(ms * 60)))
    plt.xticks(np.arange(0, 60*25, 3*60))
    plt.gca().xaxis.set_major_formatter(formatter)

    # Plot vertical lines
    def plot_vertical_line(hours, minutes, color):
        plt.axvline(x=hours*60+minutes, color=color, linestyle='--')
        plt.text(hours*60+minutes, plt.ylim()[1], f'{hours:02d}:{minutes:02d}', color=color, ha='center', va='bottom')

    plot_vertical_line(7, 50, 'red')
    plot_vertical_line(17, 30, 'red')
    plot_vertical_line(5, 20, 'green')
    plot_vertical_line(13, 10, 'green')
    plot_vertical_line(21, 50, 'green')

    plt.savefig(os.path.join(data_dir, 'speed_24_hours.png'), dpi=300)
    plt.clf()

if __name__ == "__main__":
    map_average_speed()
    plot_map_connections()
    map_speeds_at_time(7, 50)
    map_speeds_at_time(17, 30)
    map_speeds_at_time(5, 20)
    map_speeds_at_time(13, 10)
    map_speeds_at_time(21, 50)
    plot_speed_24_hours()
    animate_map_values(avg_day_speeds, "animation.mp4")