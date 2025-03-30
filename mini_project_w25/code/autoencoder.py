import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        """
        Encoder network for the autoencoder.
        
        Args:
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR10)
            latent_dim: Dimension of the latent space (128 as per requirements)
        """
        super(Encoder, self).__init__()
        
        # For MNIST (28x28) or CIFAR10 (32x32), we'll use a similar architecture
        # with adjustments for input size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size of the flattened features
        # For MNIST: 28x28 -> 14x14 -> 7x7 -> 4x4 (roughly)
        # For CIFAR10: 32x32 -> 16x16 -> 8x8 -> 4x4
        if in_channels == 1:  # MNIST
            self.fc_input_size = 128 * 4 * 4
        else:  # CIFAR10
            self.fc_input_size = 128 * 4 * 4
        
        # Fully connected layer to latent space
        self.fc = nn.Linear(self.fc_input_size, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        """
        Decoder network for the autoencoder.
        
        Args:
            latent_dim: Dimension of the latent space
            out_channels: Number of output channels (1 for MNIST, 3 for CIFAR10)
        """
        super(Decoder, self).__init__()
        
        # For MNIST or CIFAR10, we'll use a similar architecture
        if out_channels == 1:  # MNIST
            self.initial_size = 4  # Starting spatial size
        else:  # CIFAR10
            self.initial_size = 4
        
        # Fully connected layer from latent space to initial features
        self.fc_input_size = 128 * self.initial_size * self.initial_size
        self.fc = nn.Linear(latent_dim, self.fc_input_size)
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, self.initial_size, self.initial_size)  # Reshape
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))  # Output in range [-1, 1]
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Complete autoencoder model.
        
        Args:
            encoder: Encoder network
            decoder: Decoder network
        """
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        """
        Classifier that uses the latent representation from the encoder.
        
        Args:
            latent_dim: Dimension of the latent space
            num_classes: Number of output classes (10 for both MNIST and CIFAR10)
        """
        super(Classifier, self).__init__()
        
        # Simple MLP classifier
        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x