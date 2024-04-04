import torch
import torch.nn as nn
import numpy as np


def get_position_encodings(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


class LearnableSinusoidalEmbedding(nn.Module):
    def __init__(self, T):
        super(LearnableSinusoidalEmbedding, self).__init__()
        self.linear_1 = nn.Linear(50, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.position_encodings = torch.tensor(get_position_encodings(seq_len=T, d=50), dtype=torch.float)
        self.relu = nn.ReLU()

    def forward(self, t):
        position_encoding = self.position_encodings[t]
        x = self.relu(self.linear_1(position_encoding))
        x = self.linear_2(x)
        return x


class ConditionalDenseLayer(nn.Module):
    def __init__(self, T, last=False):
        super(ConditionalDenseLayer, self).__init__()
        self.linear = nn.LazyLinear(128)
        self.sinusoidal_embedding = LearnableSinusoidalEmbedding(T)
        self.relu = nn.ReLU()
        self.last = last
        if last:
            self.output_layer = nn.Linear(128, 2)

    def forward(self, x, t):
        x = self.linear(x)
        position_encoding = self.sinusoidal_embedding(t)
        x = self.relu(x + position_encoding)
        if self.last:
            x = self.output_layer(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, T):
        super(DiffusionModel, self).__init__()
        self.conditional_dense_1 = ConditionalDenseLayer(T)
        self.conditional_dense_2 = ConditionalDenseLayer(T)
        self.conditional_dense_3 = ConditionalDenseLayer(T)
        self.conditional_dense_4 = ConditionalDenseLayer(T, last=True)

    def forward(self, x, t):
        x = self.conditional_dense_1(x, t)
        x = self.conditional_dense_2(x, t)
        x = self.conditional_dense_3(x, t)
        x = self.conditional_dense_4(x, t)
        return x


if __name__ == '__main__':
    model = DiffusionModel(1000)
    x = torch.randn((10, 2))
    t = torch.randint(low=0, high=1000, size=(10,))
    print(model(x, t).shape)
