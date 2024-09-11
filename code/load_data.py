import os
import pickle
import torch
from torchvision.transforms import Compose, ToTensor, Grayscale
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ==================================================================================================
# Define transformations
# ==================================================================================================

simple_transform = Compose([
    
    ToTensor()
])

fixation_transform = Compose([
    Grayscale(),
    
    ToTensor()
])

# ==================================================================================================
# Loading images from a directory
# ==================================================================================================

def load_images_from_dir(directory, transform):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)

# Function to check if the folder is empty
def is_folder_empty(folder):
    return not any(os.scandir(folder))

def data_download():
    # Correct the data directory path based on the given structure
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    cache_file = "./data_cache.pkl"

    if os.path.exists(cache_file):
        print("Loading dataset from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Loading dataset...")


    # Load the data
    data = {
        'train': {
            'image': torch.stack([simple_transform(img) for img, _ in ImageFolder(root=data_dir + '/cv2_training_data/images')]),
            'fixations': torch.stack([fixation_transform(img) for img, _ in ImageFolder(root=data_dir + '/cv2_training_data/fixations')])
        },
        'validation': {
            'image': torch.stack([simple_transform(img) for img, _ in ImageFolder(root=data_dir + '/cv2_validation_data/images')]),
            'fixations': torch.stack([fixation_transform(img) for img, _ in ImageFolder(root=data_dir + '/cv2_validation_data/fixations')])
        }
    }

    print("Caching the dataset...")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data

def create_dataloaders(data, batch_size, num_workers=0, pin_memory=False, persistent_workers=False):
    print("Inside create dataloaders")

    # Print shapes of data tensors for debugging
    print(f"Train images shape: {data['train']['image'].shape}")
    print(f"Train fixations shape: {data['train']['fixations'].shape}")
    print(f"Validation images shape: {data['validation']['image'].shape}")
    print(f"Validation fixations shape: {data['validation']['fixations'].shape}")

    # Creating train dataloader
    train_dataset = TensorDataset(data['train']['image'], data['train']['fixations'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Creating validation dataloader
    validation_dataset = TensorDataset(data['validation']['image'], data['validation']['fixations'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        'train': train_dataloader,
        'validation': validation_dataloader,
    }

    print("Outside dataloaders")
    return dataloaders


# ==================================================================================================
# Testing load_data function
# ==================================================================================================


def display_images(training_data):
    # Get the first batch of images and fixations
    images, fixations = next(iter(training_data))

    print("Images shape: ", images.shape)
    print("Fixations shape: ", fixations.shape)

    # Plot the first 5 images
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
    plt.show()

    # Plot the first 5 fixations
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(fixations[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    data = data_download()
    data_loaders = create_dataloaders(data, batch_size=64)
    display_images(data_loaders['train'])
