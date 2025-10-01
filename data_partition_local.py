import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import pathlib
from collections import defaultdict
from local_breast import commun_test_set

#path for mnist data set
datamnist='/Users/delio/Documents/Working_projects/Balazs/Experiments/MNIST/data_mnist'
#transform and path for brain data set
source_dir= '/Users/delio/Documents/Working_projects/Balazs/Experiments/Brain/archive'### path for brain data set
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


def partition_data(dataname, num_clients, alpha=0.5, seed=42, train_ratio=0.8, partition_type='N-IID'):
    np.random.seed(seed)
    
    if dataname == "MNIST":
        dataset = datasets.MNIST(root=datamnist, train=True, download=True, transform=transforms.ToTensor())
    elif dataname == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataname == "BRAIN":
        dataset = torchvision.datasets.ImageFolder(source_dir, transform=transform_brain)
    else:
        raise ValueError("Unsupported dataset")
    
    labels = np.array(dataset.targets)
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    client_indices = defaultdict(list)
    
    if partition_type == 'N-IID':
        indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}
        for c in range(num_classes):
            np.random.shuffle(indices_per_class[c])
            num_samples_class = len(indices_per_class[c])
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            proportions = (np.array(proportions) * num_samples_class).astype(int)
            while proportions.sum() < num_samples_class:
                proportions[np.argmax(proportions)] += 1
            while proportions.sum() > num_samples_class:
                proportions[np.argmax(proportions)] -= 1
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices[i].extend(indices_per_class[c][start:end])
                start = end
    else:  # IID case
        shuffled_indices = np.random.permutation(num_samples)
        split_sizes = [num_samples // num_clients] * num_clients
        for i in range(num_samples % num_clients):
            split_sizes[i] += 1
        client_splits = np.split(shuffled_indices, np.cumsum(split_sizes)[:-1])
        for i, split in enumerate(client_splits):
            client_indices[i] = split.tolist()
    
    client_train_test = {}
    for client, indices in client_indices.items():
        train_size = int(len(indices) * train_ratio)
        train_indices, test_indices = indices[:train_size], indices[train_size:]
        client_train_test[client] = {"train": train_indices, "test": test_indices}
    
    return client_train_test, dataset

def data_for_clients(data_name, num_clients, alpha=0.5, train_ratio=0.8, partition_type='N-IID'):
    client_partitions, dataset = partition_data(data_name, num_clients, alpha, train_ratio=train_ratio, partition_type=partition_type)
    client_data = {}
    
    for client, indices in client_partitions.items():
        train_loader = DataLoader(Subset(dataset, indices['train']), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, indices['test']), batch_size=32, shuffle=False)
        client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    
    return client_data



# def commun_test_set(data_name):
#     """
#     just conver test into data loader depending on the chosen data set
#     """
#     if data_name=="MNIST":
#         dataset=datasets.MNIST(root=datamnist, train=False, download=False, transform=transforms.ToTensor())
#     elif data_name=="CIFAR10":
#         transform = transforms.Compose([transforms.ToTensor()])
#         dataset=datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
#     elif data_name=="BRAIN":
#         # Define an object of the custom dataset for the train and validation.
#         dataset = torchvision.datasets.ImageFolder(source_dir.joinpath("Testing"), transform=transform_brain)
#     else:
#         raise ValueError("Not support data set")
#     combined_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     return combined_dataloader

if __name__ == "__main__":
    num_clients = 6
    partition_type = "IID"  # Change to "N-IID" for non-IID partitioning
    client_data = data_for_clients("BRAIN", num_clients, partition_type=partition_type)
    common=commun_test_set(client_data)
    print(len(common.dataset))
    # for client, loaders in client_data.items():
    #     print(f"Client {client} - Train size: {len(loaders['train_loader'].dataset)}, Test size: {len(loaders['test_loader'].dataset)}")
