import torch
import numpy as np
import torch.nn as nn
import math
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split,DataLoader, Subset, TensorDataset,Dataset
import random
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt
from data_stroke import add_randomized_response_noise


np.random.seed(42)
torch.manual_seed(42)


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


dataset = torchvision.datasets.ImageFolder(source_dir, transform=transform_brain)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for validation

# Split dataset into training and validation
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


class set_to_dataset(Dataset):
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label

# def subset_to_tensordataset(original_dataset, subset):
#     """
#     Convert a Subset to a TensorDataset.
    
#     Args:
#         original_dataset (Dataset): The original dataset (e.g., CIFAR-10).
#         subset (Subset): A subset of the original dataset.
        
#     Returns:
#         TensorDataset: A TensorDataset containing data and labels from the subset.
#     """
#     indices = subset.indices
#     # Get the data and targets using the subset indices
#     data = torch.stack([original_dataset[i][0] for i in indices])  # Collect all data in the subset
#     targets = torch.tensor([original_dataset[i][1] for i in indices])  # Collect all labels in the subset
    
#     # Create a TensorDataset from the data and targets
#     return TensorDataset(data, targets)

train=set_to_dataset(dataset, train_dataset)

class RandomizedResponseDataset(Dataset):
    """
    Custom dataset wrapper that applies the Randomized Response (RR) mechanism to the labels.
    
    Args:
        dataset (Dataset): Original dataset (e.g., CIFAR-10).
        noise_rate (float): The probability of noise (e.g., 0.5 means 50% chance).
    """
    def __init__(self, dataset, noise_rate=0.5):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = 4  # Number of classes in the dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply the randomized response noise to the label
        noisy_label = self._apply_randomized_response(label)

        return image, noisy_label

    def _apply_randomized_response(self, label):
        """
        Applies the Randomized Response (RR) mechanism to the label.

        Args:
            label (int): Original label of the image.

        Returns:
            noisy_label (int): The noisy label after applying RR.
        """
        if random.random() < self.noise_rate:
            # With probability (1 - noise_rate), the label is kept unchanged
            if random.random() < self.noise_rate/(self.num_classes-1): 
                new_label = label
                while new_label == label:
                    new_label = random.choice(range(self.num_classes))
                return new_label
            else:
                # Otherwise, choose a random incorrect label
                return label
        else:
            # No noise, return the original label
            return label


def partition_data(dataname, num_clients, alpha=0.5, partition_type='N-IID',seed=42):
    np.random.seed(seed)
    if dataname=="BRAIN":
        targets = []
        for _, target in train:
            targets.append(target)
        labels = np.array(targets)
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
        elif partition_type == 'IID':
            # IID partitioning: Shuffle and equally distribute data
            indices = np.random.permutation(num_samples)
            samples_per_client = num_samples // num_clients
            for i in range(num_clients):
                client_indices[i] = indices[i * samples_per_client: (i + 1) * samples_per_client].tolist()
    else:
        raise ValueError("Not support data set")
    return client_indices


def data_for_clients_brain(data_name, num_clients, alpha=0.5, partition_type='N-IID'):
    client_partitions= partition_data(data_name, num_clients, alpha, partition_type=partition_type)
    client_data = {}
    
    if partition_type=="N-IID":
        for client, indices in client_partitions.items():
            client_datos=Subset(train,indices)
            cl_datos=set_to_dataset(train,client_datos)
            train_size_client = int(0.8 * len(cl_datos))  # 80% for training
            test_size_client = len(cl_datos) - train_size_client 
            entrenar, probar= random_split(cl_datos,[train_size_client,test_size_client])
            train_loader = DataLoader(entrenar,batch_size=32, shuffle=True)
            test_loader = DataLoader(probar,batch_size=32, shuffle=True)
            client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    elif partition_type=="IID":
        noise_levels = [(i+1)/(num_clients+1) for i in range(num_clients)]
        for client, indices in client_partitions.items():
            client_datos=Subset(train,indices)
            cl_datos=set_to_dataset(train,client_datos)
            noisy_cliente=RandomizedResponseDataset(cl_datos,noise_rate=noise_levels[client])
            train_size_client = int(0.8 * len(noisy_cliente))  # 80% for training
            test_size_client = len(noisy_cliente) - train_size_client 
            entrenar, probar= random_split(noisy_cliente,[train_size_client,test_size_client])
            train_loader = DataLoader(entrenar,batch_size=32, shuffle=True)
            test_loader = DataLoader(probar,batch_size=32, shuffle=True)
            client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    
    return client_data


def commun_test_set_brain():
    test=set_to_dataset(dataset, test_dataset)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)
    return test_loader


if __name__ == "__main__":
    print(len(train))
    data_name="BRAIN"
    partition="N-IID"
    alpha=0.5
    num_clients=6
    set=data_for_clients_brain(data_name,num_clients,alpha,partition)
    print(len(set[0]["train_loader"].dataset))
    print(len(set[1]["train_loader"].dataset))
    print(len(set[2]["train_loader"].dataset))
    print(len(set[3]["train_loader"].dataset))
    print(len(set[4]["train_loader"].dataset))
    print(len(set[5]["train_loader"].dataset))
    print(len(set[0]["test_loader"].dataset))
    print(len(set[1]["test_loader"].dataset))
    print(len(set[2]["test_loader"].dataset))
    print(len(set[3]["test_loader"].dataset))
    print(len(set[4]["test_loader"].dataset))
    print(len(set[5]["test_loader"].dataset))
    