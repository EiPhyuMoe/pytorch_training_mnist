"""
file : network.py

author : EPM
cdate : Monday December 2nd 2024
mdate : Monday December 2nd 2024
copyright: 2024 GlobalWalkers.inc. All rights reserved.
"""

import torch.nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        output = self.flatten(x)
        output = self.sequential_stack(output)
        return output


if __name__ == "__main__":
    net = NeuralNetwork()
    input = torch.rand(1, 28, 28)
    output = net(input)
    print(output)
