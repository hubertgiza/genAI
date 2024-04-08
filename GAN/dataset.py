import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
from torchvision import transforms
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


class PokemonDataset(Dataset):
    def __init__(self, root_path: str):
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        available_images = list(filter(lambda x: x.endswith(".png"), os.listdir(root_path)))
        self.images = []
        self.images2 = []
        for image in available_images:
            try:
                path = f"{root_path}/{image}"
                curr_img = transform(Image.open(path).convert("RGB"))
                self.images.append(curr_img)
                # self.images2.append(cv2.imread(path))
            except PIL.UnidentifiedImageError:
                continue
        # print("ok")

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = PokemonDataset("sprites/sprites/pokemon")
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Decide which device we want to run on
    device = torch.device("cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
