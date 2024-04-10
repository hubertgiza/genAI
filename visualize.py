import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.utils as vutils

from GAN.generator import Generator

if __name__ == '__main__':
    device = torch.device("cuda")
    noise_size = 100
    feature_map_size = 64
    image_size = 64

    netG = Generator(noise_size, feature_map_size).to(device)
    checkpoint = torch.load('GAN/checkpoints/checkpoint_200/checkpoint.pth')
    netG.load_state_dict(checkpoint['generator_state_dict'])
    noise = torch.randn(16, noise_size, 1, 1, device=device)
    output = netG(noise).detach().cpu()
    image_array = ((output.permute(0, 2, 3, 1).numpy() + 1) * 128).astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(output[:16], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
