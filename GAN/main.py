import math

import torch
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import torch.autograd as autograd

from matplotlib import pyplot as plt
from GAN.dataset import PokemonDataset
from GAN.discriminator import Discriminator
from GAN.generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms

LAMBDA = 10  # Gradient penalty lambda coefficient

fixed_noise = torch.randn(64, 100, 1, 1, device=torch.device("cuda"), dtype=torch.float)
samples_path = os.path.join('./samples', "pokemon")


def generate_imgs(netG, epoch):
    netG.eval()
    fake_imgs = netG(fixed_noise)
    fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(fixed_noise.shape[0] ** 0.5))
    vutils.save_image(fake_imgs_, os.path.join(samples_path, 'sample_' + str(epoch) + '.png'))
    netG.train()


def get_infinite_batches(data_loader):
    while True:
        for images in data_loader:
            yield images


def calc_gradient_penalty(netD, real_data: torch.Tensor, fake_data: torch.Tensor):
    m = real_data.shape[0]
    epsilon = torch.rand(m, 1, 1, 1).to(device)

    interpolates = epsilon * real_data + ((1 - epsilon) * fake_data)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True,
                              retain_graph=True)[0]
    gradients = gradients.reshape((m, -1))
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def create_checkpoint(directory_path: str, file_name: str, checkpoint_dict: dict):
    path = os.path.join(directory_path, file_name)
    os.makedirs(directory_path, exist_ok=True)
    torch.save(checkpoint_dict, path)


if __name__ == '__main__':
    device = torch.device("cuda")
    noise_size = 100
    feature_map_size = 64
    image_size = 64

    netG = Generator(noise_size, feature_map_size).to(device)
    netD = Discriminator(3, feature_map_size).to(device)

    lr = 0.0001
    betas = (0.0, 0.9)
    weight_decay = 2e-5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    batch_size = 64
    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = PokemonDataset("sprites/sprites/pokemon", data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_dataloader = get_infinite_batches(dataloader)

    num_epochs = 50000
    n_critic = 5
    for epoch in range(num_epochs):
        ############################
        # (1) Train Discriminator(critic) n_critic times
        ###########################
        for iter_critic in range(n_critic):
            real_data = next(infinite_dataloader).to(device)
            if real_data.size()[0] != batch_size:
                continue

            netD.zero_grad()
            noise = torch.randn(batch_size, noise_size, 1, 1, dtype=torch.float, device=device)
            fake_data = netG(noise)

            error_fake = netD(fake_data.detach()).mean()
            error_real = netD(real_data.detach()).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)
            d_loss = -(error_real - error_fake) + gradient_penalty

            d_loss.backward()
            optimizerD.step()

        ############################
        # (2) Train Generator once
        ###########################
        netG.zero_grad()
        noise = torch.randn(batch_size, noise_size, 1, 1, dtype=torch.float, device=device)
        fake_data = netG(noise)

        error_fake = netD(fake_data).mean()
        g_loss = - error_fake
        g_loss.backward()
        optimizerG.step()

        print(f"Epoch: {epoch}",
              f" Critic loss: {d_loss.item()}",
              f" Generator loss: {g_loss.item()}",
              )
        if epoch % 200 == 0:
            directory_name = f"checkpoints/checkpoint_{epoch}"
            filename = "checkpoint.pth"
            checkpoint_dict = {
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict()
            }
            create_checkpoint(directory_name, filename, checkpoint_dict)
            generate_imgs(netG, epoch)
