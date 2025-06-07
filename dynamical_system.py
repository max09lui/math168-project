import math
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

import datetime
import os
from net import LinearDynamicalSystem, LogisticDynamicalSystem

from networkx_graph import weight_matrix, speed_matrix
from traffic_animation import animate_map_values, avg_day_speed_matrix

# timestamp subdir
save_dir = os.path.join("dynamical_system", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
num_epochs = 50000
args = parser.parse_args()
checkpoint_path = "dynamical_system/linear_2023-12-03_19-54-37/checkpoint.pth"
# checkpoint_path = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
N = 228
A = weight_matrix(N, weighted=True)
A = torch.Tensor(A, device=device)
V = speed_matrix(N)
V = torch.Tensor(V, device=device)
DT = 5 # 5 minutes
NUM_TIMES = V.shape[0]
NUM_NODES = V.shape[1]

# trainset = torch.utils.data.TensorDataset(X, Y)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# Model
print('==> Building model..')

def simulate_matrix(net):
    with torch.no_grad():
        # repeatedly feed the output back into the network
        # to get a full day of simulated traffic
        V_0 = V[0]
        V_sim = [V_0]
        for i in range(288-1):
            V_0 = torch.stack((V_0, torch.zeros(NUM_NODES, device=device)), dim=0)
            V_0 = net(V_0)
            V_sim.append(V_0[0])
            V_0 = V_0[0]
        V_sim = torch.stack(V_sim)
    return V_sim

# V_sim = simulate_matrix(net)
# animate_map_values(V_sim, "dynamical_system_epoch_0.mp4")

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)
    if 'model_name' not in checkpoint:
        net = LinearDynamicalSystem(A, timestep=DT)
    elif checkpoint['model_name'] == 'LinearDynamicalSystem':
        net = LinearDynamicalSystem(A, timestep=DT)
    elif checkpoint['model_name'] == 'LogisticDynamicalSystem':
        net = LogisticDynamicalSystem(A, timestep=DT)
    else:
        raise ValueError("Unknown model name")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
else:
    net = LogisticDynamicalSystem(A, timestep=DT)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    start_epoch = 0
    train_losses = []

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def plot_loss(train_losses):
    plt.semilogy(train_losses)
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)
    plt.clf()

def plot_weights(X, name, time):
    X = X.detach().cpu().numpy()
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title("{} data at {}".format(name, time))
    plt.savefig(os.path.join(save_dir, "{}_data_{}.png".format(name, time)), dpi=300)
    plt.clf()

# Training
def train(epoch):
    # print(net.weight[0])
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += loss.item()
    outputs = net(V)
    targets = V[1:]
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    batch_idx = 0
    # print((1 - torch.eye(A.shape[0]))[0])
    # print(A[0])
    # print((A * (1 - torch.eye(A.shape[0]))))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_loss = train_loss/(batch_idx+1)
    print('[%s] Training -- Loss: %.8e' % (timestamp, avg_loss))

    return avg_loss
    # plot the weight matrix using matplotlib

plot_weights(net.weight, "weight", "start")
plot_weights(A, "adjacency", "start")
plot_weights(V.T, "speed", "start")
plot_weights(torch.tensor(avg_day_speed_matrix(228).T), "avg_speed", "start")
for epoch in range(start_epoch, start_epoch + num_epochs):
    loss = train(epoch)
    train_losses.append(loss)
    # scheduler.step()
torch.save({
            'epoch': epoch,
            'model_name': net.__class__.__name__,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            }, os.path.join(save_dir, 'checkpoint.pth'))
plot_weights(net.weight, "weight", "final")
plot_weights(A, "adjacency", "final")
plot_weights(V.T, "speed", "final")

# V_sim = simulate_matrix(net)
# animate_map_values(V_sim, f"dynamical_system_final.mp4")

plot_loss(train_losses)