import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from GAN.dataset import PokemonDataset
from GAN.discriminator import Discriminator
from GAN.generator import Generator
from torch.utils.data import DataLoader


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != "ConvTranspose2DBlock":
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    device = torch.device("cuda")
    noise_size = 100
    feature_map_size = 96
    image_size = 96

    netG = Generator(noise_size, feature_map_size).to(device)
    # netG.apply(weights_init)

    netD = Discriminator(noise_size, feature_map_size).to(device)
    # netD.apply(weights_init)

    criterion = nn.BCELoss()

    real_label = 0.9
    fake_label = 0.1
    batch_size = 16
    lr = 0.0002
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-4)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # Training Loop
    num_epochs = 500
    # Lists to keep track of progress
    dataset = PokemonDataset("sprites/sprites/pokemon")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader),
                            desc=f"Epoch: {epoch} Loss Discriminator: Loss Generator: ",
                            # position=0,
                            # leave=True
                            )
        curr_g_losses = []
        curr_d_losses = []
        n_critic = 5
        epochs = 500
        for data in dataloader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_data = data.to(device)
            label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake_data = netG(noise).detach()
            labels_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            both_data = torch.concatenate((real_data, fake_data))
            both_labels = torch.concatenate((label_real, labels_fake))

            output = netD(both_data).view(-1)
            errD = criterion(output, both_labels)
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            new_noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            new_fake_data = netG(new_noise)
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(new_fake_data).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label_real)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()

            curr_g_losses.append(errG.item())
            curr_d_losses.append(errD.item())

            mean_g_loss = np.mean(curr_g_losses)
            mean_d_loss = np.mean(curr_d_losses)
            progress_bar.set_description(
                f"Epoch: {epoch} Loss Generator: {mean_g_loss:.3f} Loss Discriminator: {mean_d_loss:.3f}")

            progress_bar.update(1)
        if epoch % 50 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict()
            }, f'checkpoint_{epoch}.pth')

    # model = YourModel()

    # Instantiate Adam optimizer
    # optimizer = optim.Adam(netG.parameters(), lr=0.001)

    # Load model and optimizer
    # checkpoint = torch.load('checkpoint.pth')
    # netG.load_state_dict(checkpoint['generator_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # noise = torch.randn(16, noise_size, 1, 1, device=device)
    # output = netG(noise).detach().cpu()
    # image_array = ((output.permute(0, 2, 3, 1).numpy() + 1) * 128).astype(np.uint8)
    # plt.plot([1],[1])
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(output[:16], padding=2, normalize=True), (1, 2, 0)))
    # plt.show()
