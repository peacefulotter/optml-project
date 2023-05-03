
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def from_sequential():
        layers = nn.Sequential(
            nn.Linear(6, 1),
            nn.Sigmoid(),
            nn.Linear(3, 5),
            nn.Sigmoid()
        )
        return Network(layers)