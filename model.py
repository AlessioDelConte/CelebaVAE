import math

import torch
from torch import nn

init_channels = 16  # initial number of filters
image_channels = 3  # color channels


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, cnn_channels_size, input_size, latent_dim=20, kernel_size=3, stride=2, padding=1):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        reversed_cnn_channel_size = list(reversed(cnn_channels_size))

        def get_out_size(size): return math.floor((size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

        out_size = input_size
        for i in range(len(cnn_channels_size) - 1):
            out_size = get_out_size(out_size)

        self.cnn_enc_layers = nn.ModuleList([
                nn.Sequential(
                        nn.Conv2d(i, j, kernel_size, stride, padding),
                        # nn.BatchNorm2d(j),
                        nn.ReLU(True)
                ) for i, j in zip(cnn_channels_size[:-1], cnn_channels_size[1:])
        ])
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_enc = nn.Linear(cnn_channels_size[-1] * out_size ** 2, latent_dim * 2)

        self.fc_dec = nn.Linear(latent_dim, cnn_channels_size[-1] * out_size ** 2)
        self.unflatten = nn.Unflatten(1, (cnn_channels_size[-1], out_size, out_size))
        self.cnn_dec_layers = nn.ModuleList([
                nn.Sequential(
                        nn.ConvTranspose2d(i, j, kernel_size, stride, padding, padding),
                        # nn.BatchNorm2d(j),
                        nn.ReLU(True) if j != 3 else nn.Sigmoid()
                ) for i, j in zip(reversed_cnn_channel_size[:-1], reversed_cnn_channel_size[1:])
        ])

    def reparameterize(self, mu, log_var):
        """
        :param mu:  mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        # cnn encoder
        for layer in self.cnn_enc_layers:
            x = layer(x)
        # flat features for fully connected layers
        x = self.flatten(x)
        x = self.fc_enc(x)
        # fully-connected encoder layer
        mu = x[:, :self.latent_dim]
        log_var = x[:, self.latent_dim:]
        return mu, log_var

    def decode(self, z):
        z = self.fc_dec(z)
        # reshape the features to prepare con cnn layers
        z = self.unflatten(z)
        # cnn decoder layers
        for layer in self.cnn_dec_layers:
            z = layer(z)
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)

        # get the latent vector through re-parameterization
        z = self.reparameterize(mu, log_var)

        reconstruction = self.decode(z)

        return reconstruction, mu, log_var

    def get_latent(self, x):
        x = x.reshape(1, 3, 128, 128)
        return self.encode(x)
