import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.metrics import roc_auc_score

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs=14, n_hidden_neurons=20, n_outputs=2):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, n_hidden_neurons)
        self.layer2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.layer3 = nn.Linear(n_hidden_neurons, n_outputs)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x