#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ## Cell 1: Import libraries and set constants

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
from tqdm import tqdm
from utils import plot_tsne

# Set constants
NUM_CLASSES = 10
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS_AE = 30
EPOCHS_CLS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42

# Set device - FIXED: Removed MPS check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
freeze_seeds(SEED)


# In[2]:


# ## Cell 2: Define the Encoder

# In[2]:


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        """
        Encoder network for the autoencoder.
        
        Args:
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR10)
            latent_dim: Dimension of the latent space (128 as per requirements)
        """
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size of the flattened features
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


# In[3]:


# ## Cell 3: Define the Decoder

# In[3]:


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        """
        Decoder network for the autoencoder.
        """
        super(Decoder, self).__init__()
        
        # Same initial settings
        self.initial_size = 4
        self.fc_input_size = 128 * self.initial_size * self.initial_size
        self.fc = nn.Linear(latent_dim, self.fc_input_size)
        
        # First two layers remain the same for both datasets
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Precise dimensions for MNIST vs CIFAR10
        if out_channels == 1:  # MNIST needs 28×28 output
            # This will output exactly 28×28
            self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=1, stride=1)
            self.final_crop = True
        else:  # CIFAR10 needs 32×32 output
            self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
            self.final_crop = False
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, self.initial_size, self.initial_size)  # 4×4
        x = F.relu(self.bn1(self.deconv1(x)))  # 8×8
        x = F.relu(self.bn2(self.deconv2(x)))  # 16×16
        
        if self.final_crop:  # MNIST case
            # First upsample to 32×32
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            # Then apply the final convolution to keep channel dimensions
            x = torch.tanh(self.deconv3(x))  # Still 28×28
        else:  # CIFAR10 case
            x = torch.tanh(self.deconv3(x))  # 32×32
            
        return x


# In[4]:


# ## Cell 4: Define the Autoencoder

# In[4]:


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


# In[5]:


# ## Cell 5: Define the Classifier

# In[5]:


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


# In[6]:


# ## Cell 6: Training and Evaluation Functions

# In[6]:


def train_autoencoder(model, train_loader, val_loader, num_epochs, device, lr=1e-3, weight_decay=1e-5):
    """
    Train the autoencoder model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()
    
    model.train()
    mse_losses = []
    mae_losses = []
    val_mse_losses = []
    val_mae_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(data)
            mse_loss = mse_criterion(reconstructed, data)
            
            # Calculate MAE for reporting (not used for optimization)
            mae_loss = torch.mean(torch.abs(reconstructed - data))
            
            # Backward pass and optimize
            mse_loss.backward()
            optimizer.step()
            
            epoch_mse_loss += mse_loss.item()
            epoch_mae_loss += mae_loss.item()
            num_batches += 1
        
        avg_epoch_mse_loss = epoch_mse_loss / num_batches
        avg_epoch_mae_loss = epoch_mae_loss / num_batches
        mse_losses.append(avg_epoch_mse_loss)
        mae_losses.append(avg_epoch_mae_loss)
        
        # Validation
        model.eval()
        val_epoch_mse_loss = 0.0
        val_epoch_mae_loss = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                val_reconstructed, _ = model(val_data)
                
                val_mse_loss = mse_criterion(val_reconstructed, val_data)
                val_mae_loss = torch.mean(torch.abs(val_reconstructed - val_data))
                
                val_epoch_mse_loss += val_mse_loss.item()
                val_epoch_mae_loss += val_mae_loss.item()
                val_num_batches += 1
        
        avg_val_epoch_mse_loss = val_epoch_mse_loss / val_num_batches
        avg_val_epoch_mae_loss = val_epoch_mae_loss / val_num_batches
        val_mse_losses.append(avg_val_epoch_mse_loss)
        val_mae_losses.append(avg_val_epoch_mae_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {avg_epoch_mse_loss:.6f}, Train MAE: {avg_epoch_mae_loss:.6f}, Val MSE: {avg_val_epoch_mse_loss:.6f}, Val MAE: {avg_val_epoch_mae_loss:.6f}")
    
    return model, mse_losses, mae_losses, val_mse_losses, val_mae_losses

def train_classifier(encoder, classifier, train_loader, val_loader, num_epochs, device, lr=1e-3, weight_decay=1e-5):
    """
    Train the classifier using the encoder's latent representations.
    """
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    encoder.eval()  # Freeze the encoder
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                latent = encoder(data)  # Get latent representation (no gradients)
            
            outputs = classifier(latent)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        epoch_accuracy = 100.0 * correct / total
        
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation
        classifier.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for val_data, val_targets in val_loader:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                
                val_latent = encoder(val_data)
                val_outputs = classifier(val_latent)
                val_loss = criterion(val_outputs, val_targets)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_targets.size(0)
                val_correct += (val_predicted == val_targets).sum().item()
                
                val_epoch_loss += val_loss.item()
                val_num_batches += 1
        
        avg_val_epoch_loss = val_epoch_loss / val_num_batches
        val_epoch_accuracy = 100.0 * val_correct / val_total
        
        val_losses.append(avg_val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.6f}, Train Acc: {epoch_accuracy:.2f}%, Val Loss: {avg_val_epoch_loss:.6f}, Val Acc: {val_epoch_accuracy:.2f}%")
    
    return classifier, train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_classifier(encoder, classifier, dataloader, device):
    """
    Evaluate the classifier on a test dataset.
    """
    encoder.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            
            # Get latent representation
            latent = encoder(data)
            
            # Forward pass through classifier
            outputs = classifier(latent)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy


# In[7]:


# ## Cell 7: Visualization Functions

# In[7]:


def plot_training_curve(losses, val_losses, title, ylabel, save_path=None):
    """
    Plot training and validation curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def plot_reconstructions(model, dataloader, device, num_samples=5, save_path=None):
    """
    Plot original images and their reconstructions.
    """
    model.eval()
    
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Reconstruct images
    with torch.no_grad():
        reconstructions, _ = model(images)
    
    # Move to CPU and convert to numpy for plotting
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        # Plot original
        img = np.transpose(images[i], (1, 2, 0))  # CHW -> HWC
        img = (img + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        if img.shape[2] == 1:  # Grayscale
            img = img.squeeze(2)
            axes[0, i].imshow(img, cmap='gray')
        else:  # RGB
            axes[0, i].imshow(img)
        axes[0, i].set_title(f"Label: {labels[i].item()}")
        axes[0, i].axis('off')
        
        # Plot reconstruction
        recon = np.transpose(reconstructions[i], (1, 2, 0))  # CHW -> HWC
        recon = (recon + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        if recon.shape[2] == 1:  # Grayscale
            recon = recon.squeeze(2)
            axes[1, i].imshow(recon, cmap='gray')
        else:  # RGB
            axes[1, i].imshow(recon)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return images, labels

def latent_interpolation(encoder, decoder, img1, img2, steps=10, device=device, save_path=None):
    """
    Perform linear interpolation in the latent space between two images.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        img1: First image tensor
        img2: Second image tensor
        steps: Number of interpolation steps
        device: Device to use
        save_path: Path to save the interpolation plot
    """
    # Ensure images are tensors on the correct device
    img1 = img1.unsqueeze(0).to(device) if not isinstance(img1, torch.Tensor) else img1.to(device)
    img2 = img2.unsqueeze(0).to(device) if not isinstance(img2, torch.Tensor) else img2.to(device)
    
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Encode images
    with torch.no_grad():
        latent1 = encoder(img1)
        latent2 = encoder(img2)
    
    # Interpolate between latent vectors
    alphas = torch.linspace(0, 1, steps)
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            # Decode
            decoded_img = decoder(interpolated_latent)
            interpolated_images.append(decoded_img.cpu())
    
    # Visualize the interpolation
    fig, axes = plt.subplots(1, steps, figsize=(20, 4))
    
    for i, img in enumerate(interpolated_images):
        img = img.squeeze(0)
        img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        
        if img_np.shape[2] == 1:  # Grayscale
            img_np = img_np.squeeze(2)
            axes[i].imshow(img_np, cmap='gray')
        else:  # RGB
            axes[i].imshow(img_np)
        
        axes[i].axis('off')
        axes[i].set_title(f"α={alpha:.1f}" if i == 0 or i == steps-1 else "")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def custom_plot_tsne(model, dataloader, device, n_samples=1000, save_path_latent=None, save_path_image=None):
    """
    Custom t-SNE plotting function with a sample limit for efficiency.
    """
    model.eval()
    
    images_list = []
    labels_list = []
    latent_list = []
    
    # Collect data up to n_samples
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Collecting data for t-SNE"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # Get latent representation
            latent_vector = model(images)
            
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())
            
            # Check if we have enough samples
            if sum(len(l) for l in labels_list) >= n_samples:
                break
    
    # Concatenate all batches
    images = np.concatenate(images_list, axis=0)[:n_samples]
    labels = np.concatenate(labels_list, axis=0)[:n_samples]
    latent_vectors = np.concatenate(latent_list, axis=0)[:n_samples]
    
    # Plot TSNE for latent space
    print("Computing t-SNE for latent space...")
    tsne_latent = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.grid(True)
    
    if save_path_latent:
        plt.savefig(save_path_latent)
    
    plt.show()
    
    # Plot t-SNE for image space
    print("Computing t-SNE for image space...")
    images_flattened = images.reshape(images.shape[0], -1)
    tsne_image = TSNE(n_components=2, random_state=42)
    image_tsne = tsne_image.fit_transform(images_flattened)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.grid(True)
    
    if save_path_image:
        plt.savefig(save_path_image)
    
    plt.show()


# In[8]:


# ## Cell 8: Setup Dataset (MNIST)

# In[8]:


# Select dataset - set to MNIST
use_mnist = False
DATA_PATH = '/datasets/cv_datasets/data'  # Use server path

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] if use_mnist else [0.5, 0.5, 0.5], 
                        std=[0.5] if use_mnist else [0.5, 0.5, 0.5])  # Scale to [-1, 1]
])

# Load dataset
if use_mnist:
    print("Using MNIST dataset")
    train_dataset = datasets.MNIST(root=DATA_PATH, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_PATH, train=False, download=False, transform=transform)
    in_channels = 1
else:
    print("Using CIFAR10 dataset")
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=transform)
    in_channels = 3

# Split train into train and validation
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")


# In[9]:


# ## Cell 9: Initialize Models

# In[9]:


# Initialize encoder and decoder
encoder = Encoder(in_channels, LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, in_channels).to(device)
autoencoder = Autoencoder(encoder, decoder).to(device)
classifier = Classifier(LATENT_DIM, NUM_CLASSES).to(device)

# Print model architectures
print("Encoder Architecture:")
print(encoder)
print("\nDecoder Architecture:")
print(decoder)
print("\nClassifier Architecture:")
print(classifier)

# Check a sample forward pass
sample = next(iter(train_loader))[0][:1].to(device)
print(f"\nSample shape: {sample.shape}")
reconstructed, latent = autoencoder(sample)
print(f"Latent shape: {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")


# In[10]:


# ## Cell 10: Train Autoencoder (Section 1.2.1 - Self-Supervised)

# In[10]:


# Train autoencoder
print("\n--- Training Autoencoder (Self-Supervised) ---")
autoencoder, mse_losses, mae_losses, val_mse_losses, val_mae_losses = train_autoencoder(
    autoencoder, train_loader, val_loader, EPOCHS_AE, device,
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Plot training curves
plot_training_curve(
    mse_losses, val_mse_losses,
    f'Autoencoder Training MSE Loss ({("MNIST" if use_mnist else "CIFAR10")})',
    'Loss',
    save_path=f'autoencoder_mse_loss_{("mnist" if use_mnist else "cifar10")}.png'
)

plot_training_curve(
    mae_losses, val_mae_losses,
    f'Autoencoder Training MAE Loss ({("MNIST" if use_mnist else "CIFAR10")})',
    'Loss',
    save_path=f'autoencoder_mae_loss_{("mnist" if use_mnist else "cifar10")}.png'
)

# Plot reconstructions
orig_images, orig_labels = plot_reconstructions(
    autoencoder, test_loader, device, num_samples=5,
    save_path=f'reconstructions_{("mnist" if use_mnist else "cifar10")}.png'
)


# In[11]:


# ## Cell 11: Linear Interpolation (for MNIST - Section 1.2.1)

# In[11]:


if use_mnist:
    print("\n--- Performing Linear Interpolation in Latent Space ---")
    # Select 2 images from the previously visualized ones
    img1 = torch.from_numpy(orig_images[0])
    img2 = torch.from_numpy(orig_images[1])
    
    # Perform interpolation
    latent_interpolation(
        encoder, decoder, img1, img2, steps=10, device=device,
        save_path=f'interpolation_mnist.png'
    )


# In[12]:


# ## Cell 12: Train Classifier on Frozen Encoder (Section 1.2.1)

# In[12]:


# Train classifier on frozen encoder
print("\n--- Training Classifier on Encoder's Representations ---")
classifier, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier(
    encoder, classifier, train_loader, val_loader, EPOCHS_CLS, device,
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Plot training curves
plot_training_curve(
    train_losses, val_losses,
    f'Classifier Training Loss ({("MNIST" if use_mnist else "CIFAR10")})',
    'Loss',
    save_path=f'classifier_loss_{("mnist" if use_mnist else "cifar10")}.png'
)

plot_training_curve(
    train_accuracies, val_accuracies,
    f'Classifier Training Accuracy ({("MNIST" if use_mnist else "CIFAR10")})',
    'Accuracy (%)',
    save_path=f'classifier_accuracy_{("mnist" if use_mnist else "cifar10")}.png'
)


# In[13]:


# ## Cell 13: Evaluate Classifier and Visualize t-SNE (Section 1.2.1)

# In[13]:


# Evaluate on test set
print("\n--- Evaluating Classifier ---")
test_accuracy = evaluate_classifier(encoder, classifier, test_loader, device)

# Plot t-SNE visualization 
print("\n--- Computing t-SNE Visualization ---")
custom_plot_tsne(
    encoder, test_loader, device, n_samples=1000,
    save_path_latent=f'latent_tsne_{("mnist" if use_mnist else "cifar10")}_self_supervised.png',
    save_path_image=f'image_tsne_{("mnist" if use_mnist else "cifar10")}_self_supervised.png'
)

# Print summary
print("\n--- Summary (Self-Supervised Approach) ---")
print(f"Dataset: {('MNIST' if use_mnist else 'CIFAR10')}")
print(f"Latent Dimension: {LATENT_DIM}")
print(f"Autoencoder Final MSE Loss: {mse_losses[-1]:.6f}")
print(f"Autoencoder Final MAE Loss: {mae_losses[-1]:.6f}")
print(f"Classifier Train Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Classifier Validation Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Classifier Test Accuracy: {test_accuracy:.2f}%")


# In[14]:


# ## Cell 14: Classification-Guided Encoding Setup (Section 1.2.2)

# In[14]:


# Reset models for Classification-Guided approach
print("\n--- Setting up Classification-Guided Encoding ---")

# Initialize new models
encoder_cls_guided = Encoder(in_channels, LATENT_DIM).to(device)
classifier_cls_guided = Classifier(LATENT_DIM, NUM_CLASSES).to(device)

# Combined training for classification-guided encoding
def train_cls_guided(encoder, classifier, train_loader, val_loader, num_epochs, device, lr=1e-3, weight_decay=1e-5):
    """
    Train the encoder and classifier jointly for classification.
    """
    # Optimize both encoder and classifier parameters
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        encoder.train()
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through encoder and classifier
            latent = encoder(data)
            outputs = classifier(latent)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        epoch_accuracy = 100.0 * correct / total
        
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation
        encoder.eval()
        classifier.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for val_data, val_targets in val_loader:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                
                val_latent = encoder(val_data)
                val_outputs = classifier(val_latent)
                val_loss = criterion(val_outputs, val_targets)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_targets.size(0)
                val_correct += (val_predicted == val_targets).sum().item()
                
                val_epoch_loss += val_loss.item()
                val_num_batches += 1
        
        avg_val_epoch_loss = val_epoch_loss / val_num_batches
        val_epoch_accuracy = 100.0 * val_correct / val_total
        
        val_losses.append(avg_val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.6f}, Train Acc: {epoch_accuracy:.2f}%, Val Loss: {avg_val_epoch_loss:.6f}, Val Acc: {val_epoch_accuracy:.2f}%")
    
    return encoder, classifier, train_losses, train_accuracies, val_losses, val_accuracies


# In[15]:


# ## Cell 15: Train Classification-Guided Models (Section 1.2.2)

# In[15]:


# Train classification-guided models
print("\n--- Training Classification-Guided Models ---")
encoder_cls_guided, classifier_cls_guided, cls_train_losses, cls_train_accuracies, cls_val_losses, cls_val_accuracies = train_cls_guided(
    encoder_cls_guided, classifier_cls_guided, train_loader, val_loader, EPOCHS_CLS, device,
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Plot training curves
plot_training_curve(
    cls_train_losses, cls_val_losses,
    f'Classification-Guided Training Loss ({("MNIST" if use_mnist else "CIFAR10")})',
    'Loss',
    save_path=f'cls_guided_loss_{("mnist" if use_mnist else "cifar10")}.png'
)

plot_training_curve(
    cls_train_accuracies, cls_val_accuracies,
    f'Classification-Guided Training Accuracy ({("MNIST" if use_mnist else "CIFAR10")})',
    'Accuracy (%)',
    save_path=f'cls_guided_accuracy_{("mnist" if use_mnist else "cifar10")}.png'
)

# Evaluate on test set
print("\n--- Evaluating Classification-Guided Model ---")
cls_guided_test_accuracy = evaluate_classifier(encoder_cls_guided, classifier_cls_guided, test_loader, device)

# Plot t-SNE visualization
print("\n--- Computing t-SNE Visualization for Classification-Guided Encoding ---")
custom_plot_tsne(
    encoder_cls_guided, test_loader, device, n_samples=1000,
    save_path_latent=f'latent_tsne_{("mnist" if use_mnist else "cifar10")}_cls_guided.png',
    save_path_image=f'image_tsne_{("mnist" if use_mnist else "cifar10")}_cls_guided.png'
)

# Print summary
print("\n--- Summary (Classification-Guided Approach) ---")
print(f"Dataset: {('MNIST' if use_mnist else 'CIFAR10')}")
print(f"Latent Dimension: {LATENT_DIM}")
print(f"Classifier Train Accuracy: {cls_train_accuracies[-1]:.2f}%")
print(f"Classifier Validation Accuracy: {cls_val_accuracies[-1]:.2f}%")
print(f"Classifier Test Accuracy: {cls_guided_test_accuracy:.2f}%")


# In[16]:


# ## Cell 16: Structured Latent Space with Contrastive Learning (Section 1.2.3)

# In[16]:


# Implementation of structured latent space with contrastive learning
class ContrastiveAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ContrastiveAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = 0.5  # Temperature parameter for NT-Xent loss
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def contrastive_loss(self, latent_1, latent_2, labels=None):
        """
        Computes contrastive loss for pairs of latent vectors.
        If labels are provided, uses supervised contrastive loss (same class = positive pair).
        Otherwise, uses self-supervised mode (same image with different augmentations = positive pair).
        """
        batch_size = latent_1.size(0)
        
        # Normalize latent vectors
        latent_1 = F.normalize(latent_1, p=2, dim=1)
        latent_2 = F.normalize(latent_2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(latent_1, latent_2.T) / self.temperature
        
        if labels is not None:
            # Supervised contrastive loss - positive pairs are from same class
            # Create a mask for positive pairs (same class)
            labels = labels.view(-1, 1)
            mask_pos = torch.eq(labels, labels.T).float()
            
            # Remove self-pairs from positives
            mask_self = torch.eye(batch_size, device=latent_1.device)
            mask_pos = mask_pos - mask_self
            
            # Ensure each row has at least one positive
            mask_pos_sum = mask_pos.sum(1)
            mask_pos_sum = torch.where(mask_pos_sum > 0, mask_pos_sum, torch.ones_like(mask_pos_sum))
            
            # Calculate loss
            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
            loss = (mask_pos * log_prob).sum(1) / mask_pos_sum
            loss = -loss.mean()
        else:
            # Self-supervised case - assume diagonal elements are positive pairs
            # This would be appropriate if latent_1 and latent_2 are augmentations of the same images
            mask_pos = torch.eye(batch_size, device=latent_1.device)
            
            # Calculate loss
            exp_sim = torch.exp(similarity_matrix)
            log_prob = torch.diag(similarity_matrix) - torch.log(exp_sim.sum(1))
            loss = -log_prob.mean()
            
        return loss

# Define image augmentation for contrastive learning
contrastive_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] if use_mnist else [0.5, 0.5, 0.5], 
                        std=[0.5] if use_mnist else [0.5, 0.5, 0.5]),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
])

class ContrastiveDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # For MNIST and CIFAR, img is already a tensor from the original dataset
        # Convert back to PIL for additional augmentations
        # First denormalize: x = x * std + mean
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:  # MNIST
                denorm_img = img * 0.5 + 0.5  # Convert from [-1,1] to [0,1]
                pil_img = transforms.ToPILImage()(denorm_img)
            else:  # CIFAR
                denorm_img = img * 0.5 + 0.5  # Convert from [-1,1] to [0,1]
                pil_img = transforms.ToPILImage()(denorm_img)
        else:
            pil_img = img  # Already a PIL image
        
        # Create two differently augmented versions of the same image
        if self.transform:
            aug1 = self.transform(pil_img)
            aug2 = self.transform(pil_img)
            return aug1, aug2, label
        else:
            # Just convert back to tensor if no transform
            tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] if use_mnist else [0.5, 0.5, 0.5], 
                                   std=[0.5] if use_mnist else [0.5, 0.5, 0.5])
            ])
            return tensor_transform(pil_img), tensor_transform(pil_img), label
        
    def __len__(self):
        return len(self.dataset)

# Define improved image augmentation for contrastive learning
contrastive_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] if use_mnist else [0.5, 0.5, 0.5], 
                       std=[0.5] if use_mnist else [0.5, 0.5, 0.5])
])

# Create contrastive datasets and dataloaders
contrastive_train_dataset = ContrastiveDataset(train_dataset.dataset, contrastive_transform)
contrastive_train_loader = DataLoader(
    contrastive_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)


# In[17]:


# ## Cell 17: Train Contrastive Encoder (Section 1.2.3)

# In[17]:


# Initialize models for contrastive learning
encoder_contrastive = Encoder(in_channels, LATENT_DIM).to(device)
decoder_contrastive = Decoder(LATENT_DIM, in_channels).to(device)
contrastive_autoencoder = ContrastiveAutoencoder(encoder_contrastive, decoder_contrastive).to(device)

# Training function for contrastive autoencoder
def train_contrastive_autoencoder(model, train_loader, val_loader, num_epochs, device, lr=1e-3, weight_decay=1e-5, alpha=0.5):
    """
    Train the autoencoder model with contrastive learning.
    Alpha controls the balance between reconstruction and contrastive loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()
    
    model.train()
    combined_losses = []
    rec_losses = []
    con_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_con_loss = 0.0
        num_batches = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass for both augmentations
            reconstructed1, latent1 = model(img1)
            reconstructed2, latent2 = model(img2)
            
            # Reconstruction loss
            rec_loss1 = mse_criterion(reconstructed1, img1)
            rec_loss2 = mse_criterion(reconstructed2, img2)
            rec_loss = (rec_loss1 + rec_loss2) / 2
            
            # Contrastive loss
            con_loss = model.contrastive_loss(latent1, latent2, labels)
            
            # Combined loss
            loss = (1 - alpha) * rec_loss + alpha * con_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_con_loss += con_loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_rec_loss = epoch_rec_loss / num_batches
        avg_epoch_con_loss = epoch_con_loss / num_batches
        
        combined_losses.append(avg_epoch_loss)
        rec_losses.append(avg_epoch_rec_loss)
        con_losses.append(avg_epoch_con_loss)
        
        # Validation - just measure reconstruction loss
        model.eval()
        val_epoch_loss = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                val_reconstructed, _ = model(val_data)
                val_loss = mse_criterion(val_reconstructed, val_data)
                val_epoch_loss += val_loss.item()
                val_num_batches += 1
        
        avg_val_epoch_loss = val_epoch_loss / val_num_batches
        val_losses.append(avg_val_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Combined Loss: {avg_epoch_loss:.6f}, Rec Loss: {avg_epoch_rec_loss:.6f}, Con Loss: {avg_epoch_con_loss:.6f}, Val Loss: {avg_val_epoch_loss:.6f}")
    
    return model, combined_losses, rec_losses, con_losses, val_losses

# Train contrastive autoencoder
print("\n--- Training Contrastive Autoencoder ---")
contrastive_autoencoder, combined_losses, rec_losses, con_losses, val_losses = train_contrastive_autoencoder(
    contrastive_autoencoder, contrastive_train_loader, val_loader, EPOCHS_AE, device,
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, alpha=0.5  # Equal weight to reconstruction and contrastive loss
)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.plot(combined_losses, label='Combined Loss')
plt.plot(rec_losses, label='Reconstruction Loss')
plt.plot(con_losses, label='Contrastive Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f'Contrastive Autoencoder Training Losses ({("MNIST" if use_mnist else "CIFAR10")})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(f'contrastive_losses_{("mnist" if use_mnist else "cifar10")}.png')
plt.show()


# In[18]:


# ## Cell 18: Train Classifier on Contrastive Encoder (Section 1.2.3)

# In[18]:


# Initialize classifier for contrastive encoder
classifier_contrastive = Classifier(LATENT_DIM, NUM_CLASSES).to(device)

# Train classifier on frozen contrastive encoder
print("\n--- Training Classifier on Contrastive Encoder's Representations ---")
classifier_contrastive, contrastive_train_losses, contrastive_train_accuracies, contrastive_val_losses, contrastive_val_accuracies = train_classifier(
    encoder_contrastive, classifier_contrastive, train_loader, val_loader, EPOCHS_CLS, device,
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Plot training curves
plot_training_curve(
    contrastive_train_losses, contrastive_val_losses,
    f'Contrastive Classifier Training Loss ({("MNIST" if use_mnist else "CIFAR10")})',
    'Loss',
    save_path=f'contrastive_classifier_loss_{("mnist" if use_mnist else "cifar10")}.png'
)

plot_training_curve(
    contrastive_train_accuracies, contrastive_val_accuracies,
    f'Contrastive Classifier Training Accuracy ({("MNIST" if use_mnist else "CIFAR10")})',
    'Accuracy (%)',
    save_path=f'contrastive_classifier_accuracy_{("mnist" if use_mnist else "cifar10")}.png'
)

# Evaluate on test set
print("\n--- Evaluating Contrastive Classifier ---")
contrastive_test_accuracy = evaluate_classifier(encoder_contrastive, classifier_contrastive, test_loader, device)

# Plot t-SNE visualization
print("\n--- Computing t-SNE Visualization for Contrastive Encoding ---")
custom_plot_tsne(
    encoder_contrastive, test_loader, device, n_samples=1000,
    save_path_latent=f'latent_tsne_{("mnist" if use_mnist else "cifar10")}_contrastive.png',
    save_path_image=f'image_tsne_{("mnist" if use_mnist else "cifar10")}_contrastive.png'
)

# Print summary
print("\n--- Summary (Contrastive Learning Approach) ---")
print(f"Dataset: {('MNIST' if use_mnist else 'CIFAR10')}")
print(f"Latent Dimension: {LATENT_DIM}")
print(f"Classifier Train Accuracy: {contrastive_train_accuracies[-1]:.2f}%")
print(f"Classifier Validation Accuracy: {contrastive_val_accuracies[-1]:.2f}%")
print(f"Classifier Test Accuracy: {contrastive_test_accuracy:.2f}%")


# In[19]:


# ## Cell 19: Compare All Three Approaches

# In[19]:


# Compare results from all three approaches
print("\n--- Comparison of All Approaches ---")
methods = ['Self-Supervised', 'Classification-Guided', 'Contrastive Learning']
train_accs = [train_accuracies[-1], cls_train_accuracies[-1], contrastive_train_accuracies[-1]]
val_accs = [val_accuracies[-1], cls_val_accuracies[-1], contrastive_val_accuracies[-1]]
test_accs = [test_accuracy, cls_guided_test_accuracy, contrastive_test_accuracy]

# Plot comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, train_accs, width, label='Train Accuracy')
plt.bar(x, val_accs, width, label='Validation Accuracy')
plt.bar(x + width, test_accs, width, label='Test Accuracy')

plt.ylabel('Accuracy (%)')
plt.title(f'Comparison of Different Approaches ({("MNIST" if use_mnist else "CIFAR10")})')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig(f'comparison_{("mnist" if use_mnist else "cifar10")}.png')
plt.show()

# Print comparison table
print("\nAccuracy Comparison Table:")
print("-" * 80)
print(f"{'Method':<25} {'Train Accuracy':<20} {'Val Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 80)
for i, method in enumerate(methods):
    print(f"{method:<25} {train_accs[i]:<20.2f} {val_accs[i]:<20.2f} {test_accs[i]:<20.2f}")
print("-" * 80)

# End of notebook for MNIST dataset
print("\nCompleted all experiments for MNIST dataset. To run for CIFAR10, set use_mnist = False in Cell 8 and rerun.")


# In[ ]:





# In[ ]:




