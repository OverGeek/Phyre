import numpy as np
import torch
from torch import nn


class ObjLearner(nn.Module):

    def __init__(self, embedding_dim, input_dims, hidden_dim,
                 num_objects):
        super(ObjLearner, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_objects = num_objects

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.obj_extractor = CNNExtractor(
            input_dim=num_channels,
            hidden_dim=hidden_dim // 16,
            num_objects=num_objects)

        width_height = np.array(width_height)
        width_height = width_height // 5

        self.obj_encoder = EncoderMLP(
            input_dim=9747,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim)

        self.width = width_height[0]
        self.height = width_height[1]

    def run(self, obs):
        objs = self.obj_extractor(obs)
        state = self.obj_encoder(objs)

        return state

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class CNNExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_objects):
        super(CNNExtractor, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (5, 5), stride = (2, 2))
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(
            hidden_dim, hidden_dim, (5, 5), stride = (2, 2))
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5))
        self.act3 = nn.ReLU()

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.cnn3(h))
        return h


class EncoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(EncoderMLP, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Sigmoid()

    def forward(self, ins):
        h_flat = ins.view(ins.size(0), -1)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.act3(self.fc3(h))
