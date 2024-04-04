import matplotlib.pyplot as plt
import numpy as np
import torch
from model import DiffusionModel
from dataset import PointsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda")

T = 1000


def sampling(model):
    samples_shape = (10000, 2)
    alpha = model.alpha_dash[0].unsqueeze(0)
    for i in range(1, len(model.alpha_dash)):
        alphas = torch.prod(alpha[:i]).unsqueeze(0)
        alpha = torch.concatenate((alpha, model.alpha_dash[i] / alphas))
    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    beta = 1 - alpha
    x_prev = torch.randn(samples_shape, dtype=torch.float, device=device)
    model.eval()
    with torch.no_grad():
        for t in range(T - 1, 0, -1):
            z = torch.randn(samples_shape, dtype=torch.float, device=device) if t > 1 else torch.zeros(samples_shape, dtype=torch.float, device=device)
            t_vector = torch.ones((samples_shape[0]), dtype=torch.int, device=device) * t
            prediction_scale = (1 - alpha[t]) / (torch.sqrt(1 - model.alpha_dash[t]))
            x_next = (1 / torch.sqrt(alpha[t])) * (x_prev - prediction_scale * model(x_prev, t_vector)[0]) + torch.sqrt(
                beta[t]) * z
            x_prev = x_next
    return x_prev


def validate(path_to_model):
    model = torch.load(path_to_model).to(device)
    x_denoised = sampling(model).detach().cpu()
    plt.scatter(x=x_denoised[:, 0], y=x_denoised[:, 1])
    plt.show()


if __name__ == '__main__':
    validate("model_900.pth")
    # model = DiffusionModel(T).to(device)
    # train_dataset = PointsDataset("bicycle.txt")
    # train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    #
    # n_epochs = 1000
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # model.train()
    # for epoch in range(n_epochs):
    #     running_loss = 0
    #     for x_0_batch in tqdm(train_dataloader):
    #         x_0_batch = x_0_batch.to(device)
    #         number_of_samples = x_0_batch.shape[0]
    #         t = torch.randint(low=1, high=T, size=(number_of_samples,)).int()
    #         pred_epsilon, epsilon = model(x_0_batch, t)
    #         loss = criterion(pred_epsilon, epsilon)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     if epoch % 100 == 0:
    #         torch.save(model, f"model_{epoch}.pth")
    #     print(f"Step: {epoch}/{n_epochs}, loss: {running_loss / len(train_dataloader)}")
