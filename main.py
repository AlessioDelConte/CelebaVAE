from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from model import ConvVAE
from utils import image_to_vid, kld_loss, save_loss_plot, save_reconstructed_images

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.
BATCH_SIZE = 128
EPOCHS = 15

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop((128, 128))
])

celebA_data_train = torchvision.datasets.CelebA(root='~/torch-dataset', split='train', transform=transform,
                                                target_type='attr', download=False)

celebA_data_valid = torchvision.datasets.CelebA(root='~/torch-dataset', split='valid', transform=transform,
                                                target_type='attr', download=False)

train_loader = DataLoader(dataset=celebA_data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                          pin_memory=True)
valid_loader = DataLoader(dataset=celebA_data_valid, batch_size=144, shuffle=False, num_workers=8,
                          pin_memory=True)


def train(model, dataloader, criterion, optimizer):
    print("TRAIN")
    model.train()
    batch_losses = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for batch_idx, (imgs, labels) in pbar:
        # Extract data and move tensors to the selected device
        image_batch = imgs.to(device)
        # Forward pass
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(image_batch)
        bce_loss = criterion(reconstruction, image_batch)
        kld = kld_loss(mu, logvar)
        loss = bce_loss + kld
        # Backward pass
        loss.backward()
        optimizer.step()

        batch_losses.append(float(loss.data.item()))

        # Print loss
        pbar.set_postfix({'bce_loss': bce_loss.data.item(), 'KL_loss': kld.data.item(), 'min_loss': min(batch_losses)})

    return np.mean(batch_losses)


def validate(model, dataloader, criterion):
    print("VALIDATE")
    model.eval()
    batch_losses = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    with torch.no_grad():
        for batch_idx, (imgs, labels) in pbar:
            imgs = imgs.to(device)
            reconstruction, mu, logvar = model(imgs)
            bce_loss = criterion(reconstruction, imgs)
            kld = kld_loss(mu, logvar)
            loss = bce_loss + kld
            batch_losses.append(float(loss.data.item()))

            # save the last batch input and output of every epoch
            if batch_idx == int(len(dataloader)) - 2:
                recon_images = reconstruction

            # Print loss
            pbar.set_postfix(
                    {'bce_loss': bce_loss.data.item(), 'KL_loss': kld.data.item(), 'min_loss': min(batch_losses)})

    return np.mean(batch_losses), recon_images


def train_net(model, num_epochs=40):
    grid_images = []
    train_loss = []
    valid_loss = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        # Training
        train_epoch_loss = train(model, dataloader=train_loader, criterion=criterion, optimizer=optim)
        train_loss.append(train_epoch_loss)

        # Validation
        valid_epoch_loss, recon_images = validate(model, dataloader=valid_loader, criterion=criterion)
        valid_loss.append(valid_epoch_loss)
        # save the reconstructed images from the validation loop
        save_reconstructed_images(recon_images, epoch + 1)
        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu(), nrow=12)
        grid_images.append(image_grid)
        torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss'                : train_epoch_loss,
        }, "models/model.pth")

    # save the reconstructions as a .gif file
    image_to_vid(grid_images)
    # save the loss plots to disk
    save_loss_plot(train_loss, valid_loss)
    print('TRAINING COMPLETE')


if __name__ == '__main__':
    test = True
    restart = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1e-4
    model = ConvVAE([3, 64, 128, 256, 512, 728], input_size=128, latent_dim=128).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction="sum")

    if not test:
        if restart:
            checkpoint = torch.load("models/model.pth", map_location=torch.device('cpu'))

            checkpoint_model = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                name = k[7:]  # remove `module.`
                checkpoint_model[name] = v

            checkpoint["model_state_dict"] = checkpoint_model
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        train_net(model, EPOCHS)

    else:
        checkpoint = torch.load("models/model.pth", map_location=torch.device('cpu'))

        checkpoint_model = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:]  # remove `module.`
            checkpoint_model[name] = v

        checkpoint["model_state_dict"] = checkpoint_model

        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']


