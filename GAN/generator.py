import torch.nn as nn


class ConvTranspose2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTranspose2DBlock, self).__init__()
        self.conv_transpose_2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose_2d(x)
        x = self.batch_norm2d(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_size, feature_map_size):
        super(Generator, self).__init__()
        self.conv_transpose_1 = ConvTranspose2DBlock(noise_size, feature_map_size * 8, 4, 1, 0)
        self.conv_transpose_2 = ConvTranspose2DBlock(feature_map_size * 8, feature_map_size * 4, 4, 2, 1)
        self.conv_transpose_3 = ConvTranspose2DBlock(feature_map_size * 4, feature_map_size * 2, 4, 2, 1)
        self.conv_transpose_4 = ConvTranspose2DBlock(feature_map_size * 2, feature_map_size, 4, 2, 1)
        self.last_conv_transpose = nn.ConvTranspose2d(feature_map_size, 3, 4, 2, 1, bias=False)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.conv_transpose_1(x)
        x = self.conv_transpose_2(x)
        x = self.conv_transpose_3(x)
        x = self.conv_transpose_4(x)
        x = self.last_conv_transpose(x)
        x = self.output_activation(x)
        return x
