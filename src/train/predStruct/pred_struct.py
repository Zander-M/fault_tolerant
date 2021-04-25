'''
    Predict hardware structure of the robot using Multilayer Perceptron.
'''

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from faultDataset import FaultDataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(238, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x.float())
        return logits


class Mlp(nn.Module):
    def __init__(
            self,
            input_size=238,
            output_size=8,
            hidden_sizes=[512, 512, 512],
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        # here we use ModuleList so that the layers in it can be
        # detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        # initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        # init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, avg_diff = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_model", type=int, default=1000)
    parser.add_argument("--num_action", type=int, default=100)
    parser.add_argument("--joint_error", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ep", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model", type=bool, default=False)

    args = parser.parse_args()
    data = FaultDataset(joint_error=args.joint_error,
                        num_action=args.num_action,
                        num_model=args.num_model)

    print(len(data))
    random.seed(args.seed)
    test_size = int(args.test_size*len(data))
    train_size = len(data) - test_size
    print(test_size, train_size)

    test_data, train_data = random_split(
        data, [test_size, train_size])


    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    model = Mlp(238, 8, [512, 512, 512])
    # model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.ep
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    if args.save_model:
        torch.save(model.state_dict(), "predStruct.pth")
