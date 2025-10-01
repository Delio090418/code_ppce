import torch
import numpy as np
import torch.nn as nn
import math
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt

#path for mnist data set
datamnist='/Users/delio/Documents/Working projects/Balazs/Experiments/MNIST/data_mnist'
#transform and path for brain data set
source_dir= '/Users/delio/Documents/Working projects/Balazs/Experiments/Brain/archive'### path for brain data set
source_dir = pathlib.Path(source_dir)
#I have cifar10 in the local folder, so no need the path here. but in case place it here

##transfor for brain data set
transform_brain = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class NoisyDataset():
    """
    Custom dataset wrapper to apply Gaussian noise to each sample.
    """
    def __init__(self, dataset, indices, noise_std=0.1):
        self.dataset = dataset
        self.indices = indices
        self.noise_std = noise_std  # Noise level for this dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        # Apply Gaussian noise
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)  # Ensure values remain in [0,1]

        return noisy_image, label
    
def data_loader(data_set, subset, noise_std=0.1,partition_type='N-IID'):
    """
    Create a DataLoader with Gaussian noise applied to data.
    
    Args:
        data_set: Dataset object (e.g., MNIST, CIFAR-10, BRAIN)
        subset: List of indices assigned to a client
        noise_std: Standard deviation of Gaussian noise

    Returns:
        DataLoader with noisy data.
    """
    if partition_type=="IID":
        noisy_subset = NoisyDataset(data_set, subset, noise_std=noise_std)
        loader = DataLoader(noisy_subset, batch_size=32, shuffle=True)
    elif partition_type=="N-IID":
        loader=DataLoader(Subset(data_set,subset), batch_size=32, shuffle=True)
        # loader=DataLoader(Subset(data_set,subset),batch_size=32, shuffle=True)
    return loader


def partition_data(dataname, num_clients, alpha=0.5, seed=42, partition_type='N-IID'):
    """
    Partitions dataset into `num_clients` using either IID or Dirichlet-based Non-IID allocation.

    Args:
        dataname (str): Dataset name ('MNIST', 'CIFAR10', 'BRAIN').
        num_clients (int): Number of clients.
        alpha (float): Dirichlet parameter for non-IID (ignored if IID).
        seed (int): Random seed for reproducibility.
        partition_type (str): Type of partition ('IID' or 'N-IID').

    Returns:
        dict: Dictionary where keys are client indices and values are dataset indices.
    """
    np.random.seed(seed)

    if dataname=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=True, download=False, transform=ToTensor())
    elif dataname=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataname=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Training"), transform=transform_brain) 
        dataset.transform
    else:
        raise ValueError("Not support data set")

    # Get labels from dataset
    if hasattr(dataset, 'targets'):  # Works for CIFAR-10, MNIST
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):  # Some older torchvision versions
        labels = np.array(dataset.train_labels)
    else:
        raise ValueError("Dataset format not recognized, check label attribute.")

    num_classes = len(np.unique(labels))
    indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Dirichlet distribution to allocate data
    client_indices = defaultdict(list)

    if partition_type == 'N-IID':
        for c in range(num_classes):
            np.random.shuffle(indices_per_class[c])  # Shuffle indices of class `c`
            num_samples = len(indices_per_class[c])
            
            # Sample proportions for each client from Dirichlet distribution
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            
            # Convert proportions into actual data splits
            proportions = (np.array(proportions) * num_samples).astype(int)
            
            # Fix rounding issues to ensure sum equals num_samples
            while proportions.sum() < num_samples:
                proportions[np.argmax(proportions)] += 1
            while proportions.sum() > num_samples:
                proportions[np.argmax(proportions)] -= 1
            
            # Assign indices to each client
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices[i].extend(indices_per_class[c][start:end])
                start = end

    elif partition_type == 'IID':
        # IID partitioning: Shuffle and equally distribute data
        indices = np.random.permutation(num_samples)
        samples_per_client = num_samples // num_clients
        
        for i in range(num_clients):
            client_indices[i] = indices[i * samples_per_client: (i + 1) * samples_per_client].tolist()

    else:
        raise ValueError("Invalid partition type. Choose 'IID' or 'N-IID'.")
        
    return client_indices 



def data_for_client(data_name,num_clients,alpha=0.5,partition_type='N-IID'):
    """
    depending on the data set output the dataloaders for each client
    """
    if data_name=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=True, download=False, transform=ToTensor())
    elif data_name=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif data_name=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Training"), transform=transform_brain) 
    else:
        raise ValueError("Not support data set")
    dict_loaders=[]
    training=partition_data(data_name,num_clients,alpha)
    for client in range(num_clients):
        if partition_type=="N-IID":
            dict_loaders.append(data_loader(data_set=dataset, subset=training[client],partition_type=partition_type))
        elif partition_type=="IID":
            noise_levels = [i/(num_clients+1) for i in range(1,num_clients+1)]#np.linspace(0.05, 0.1, num_clients)
            dict_loaders.append(data_loader(dataset, training[client],noise_levels[client],partition_type))
    return dict_loaders

def commun_test_set(data_name):
    """
    just conver test into data loader depending on the chosen data set
    """
    if data_name=="MNIST":
        dataset=datasets.MNIST(root=datamnist, train=False, download=False, transform=ToTensor())
    elif data_name=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset=datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif data_name=="BRAIN":
        # Define an object of the custom dataset for the train and validation.
        dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Testing"), transform=transform_brain)
    else:
        raise ValueError("Not support data set")
    combined_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return combined_dataloader


if __name__ == "__main__":
    data_name="MNIST"
    partition="IID"
    alpha=0.5
    num_clients=3
    set=data_for_client(data_name,num_clients,alpha,partition)
    # Get the first batch from Client 0
    # Load a batch of images and labels for visualization
    data_iter = iter(set[1])
    images, labels = next(data_iter)

    # Convert images to numpy arrays and denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = (images.numpy().transpose((0, 2, 3, 1)) * std + mean).clip(0, 1)

    # Create a grid of images
    num_images = len(images)
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 15))
    conjun=set[1].dataset.dataset
    # Plot images with labels
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f'Label: {conjun.classes[labels[i]]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print(len(set[0].dataset))
    for data, labels in set[0]:
        print(f"Input shape: {data.shape}")   # Shape of input tensor
        print(f"Label shape: {labels.shape}") # Shape of label tensor
        break
    # print(len(set[1].dataset))
    # print("Classes:", dataset.classes)
    # print("Class to Index Mapping:", dataset.class_to_idx)
    # #print("First 5 Image Paths and Labels:", dataset.samples[:5])
    # print("First 5 Targets (Labels):", dataset.targets[:5])
    # # data=data_for_client_NoIID("MNIST",10)
    # # print(len(data[0].dataset))
