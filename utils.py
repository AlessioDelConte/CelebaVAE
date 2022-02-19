import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image


def show_tensor_as_img(tensor):
    plt.imshow(tensor.permute(1, 2, 0).cpu().numpy())
    plt.show()


def kld_loss(mu, logvar):
    """
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


to_pil_image = transforms.ToPILImage()


def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('outputs/generated_images.gif', imgs)


def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"outputs/output{epoch}.jpg", nrow=12)


def save_loss_plot(train_loss, valid_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.jpg')
    plt.show()
