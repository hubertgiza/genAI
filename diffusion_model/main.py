import matplotlib.pyplot as plt
import numpy as np
import torch

T = 1000
t = np.linspace(6, -6, T)

alpha_dash = np.array(torch.nn.Sigmoid()(torch.tensor(t)))

alpha = [alpha_dash[0]]
for i in range(1, len(alpha_dash)):
    alphas = np.product(alpha[:i])
    alpha.append(alpha_dash[i] / alphas)
alpha = np.array(alpha)
beta = 1 - alpha


def load_data(path: str):
    points = []
    with open(path, "r") as f:
        line = f.readline()
        while line:
            x, y = map(float, line.split(" "))
            points.append((x, y))
            line = f.readline()
    return np.array(points)


def forward_diffusion(x_0, t):
    return np.random.uniform(x_0 * np.sqrt(alpha_dash[t]), 1 - alpha_dash[t])


if __name__ == '__main__':
    points = load_data("bicycle.txt")
    points_1 = forward_diffusion(points, 1)
    points_10 = forward_diffusion(points, 10)
    points_100 = forward_diffusion(points, 100)
    points_500 = forward_diffusion(points, 500)
    points_1000 = forward_diffusion(points, 999)

    plt.scatter(x=points_1[:, 0], y=points_1[:, 1])
    plt.show()

    plt.scatter(x=points_10[:, 0], y=points_10[:, 1])
    plt.show()

    plt.scatter(x=points_100[:, 0], y=points_100[:, 1])
    plt.show()

    plt.scatter(x=points_500[:, 0], y=points_500[:, 1])
    plt.show()

    plt.scatter(x=points_1000[:, 0], y=points_1000[:, 1])
    plt.show()
