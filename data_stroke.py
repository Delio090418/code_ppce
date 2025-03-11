import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
#from local_breast import commun_test_set
from datastrokes_explo import X_data, y_labels
import random



np.random.seed(42)
torch.manual_seed(42)

y=np.array(y_labels)
scaler = StandardScaler()
X = scaler.fit_transform(X_data)

global_test_size=0.2
# Function to split dataset into train and test

X_train, X_test, y_train, y_test= train_test_split(
            X, y, test_size=global_test_size, stratify=y, random_state=42)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze(dim=1)


def partition_data(num_cl=3,alpha=0.5):
    """
    Partitions data into three sets with varying class distributions.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Labels (0 or 1).
    alpha (list): Dirichlet distribution parameters controlling class distribution.

    Returns:
    dict: Partitioned datasets {'set1': (X1, y1), 'set2': (X2, y2), 'set3': (X3, y3)}
    """
    # Separate indices of each class
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]

    # Generate Dirichlet proportions for partitioning
    proportions_0 = np.random.dirichlet([(20)*alpha, (20)*alpha, 1], size=len(idx_0))
    proportions_1 = np.random.dirichlet([(20)*alpha, (20)*alpha, 1][::-1], size=len(idx_1))  # Reverse alpha for class 1

    # Assign each data point to a partition
    def assign_partitions(indices, proportions):
        partitions = {0: [], 1: [], 2: []}
        for i, probs in zip(indices, proportions):
            partition = np.argmax(probs)  # Assign to the partition with the highest probability
            partitions[partition].append(i)
        return partitions

    partitions_0 = assign_partitions(idx_0, proportions_0)
    partitions_1 = assign_partitions(idx_1, proportions_1)

    # Combine partitions
    partitions = {0: partitions_0[0] + partitions_1[0],
                  1: partitions_0[1] + partitions_1[1],
                  2: partitions_0[2] + partitions_1[2]}
    return [partitions[i] for i in range(num_cl)]
   


def add_randomized_response_noise(labels, noise_rate=0.5):
    """
    Applies the Randomized Response (RR) mechanism to introduce label noise in a multi-class dataset.

    Args:
        labels (torch.Tensor): Tensor of original labels (shape: [N])
        noise_rate (float): Probability of applying noise (e.g., 0.5 means 50% chance)

    Returns:
        torch.Tensor: Noisy labels with randomized response applied.
    """
    unique_labels = labels.unique(sorted=True)  # Get unique class labels
    num_classes = len(unique_labels)

    noisy_labels = labels.clone()  # Copy the labels to modify

    for i in range(len(labels)):
        if random.random() < noise_rate:  # Apply noise with probability 'noise_rate'
            if random.random() < noise_rate/(num_classes-1):  
                new_label = labels[i]
                while new_label == labels[i]:  # Ensure a different label is chosen
                    new_label = random.choice(unique_labels.tolist())
                noisy_labels[i] = new_label
            else:
                noisy_labels[i] = labels[i]
    return noisy_labels


def partition_data_stroke(num_clients=3, alpha=0.5, local_test_size=0.2, partition_type='N-IID'):
    """
    Partitions data among clients using IID or Non-IID partitioning, adds Gaussian noise to local training data,
    and creates a local test set for each client.

    Args:
        num_clients (int): Number of clients.
        alpha (float): Dirichlet parameter (smaller -> more non-iid).
        test_size (float): Fraction of total test data.
        local_test_size (float): Fraction of each client's local test set.
        noise_levels (list): List of noise standard deviations for each client.
        partition_type (str): Type of partitioning ('IID' or 'N-IID').

    Returns:
        train_dataloaders (list): List of DataLoaders for each client's training set.
        test_dataloaders (list): List of DataLoaders for each client's local test set.
    """
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze(dim=1)
    
    num_samples = len(X_train_tensor)
    

    
    client_train_dataloaders = []
    client_test_dataloaders = []
    
    if partition_type == 'N-IID':
        client_indices=partition_data(num_cl=num_clients,alpha=alpha)
    
    elif partition_type == 'IID':
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)  # Evenly distribute indices
    
    else:
        raise ValueError("Invalid partition_type. Choose either 'IID' or 'N-IID'.")
    
    # Split each client's data into local train/test
    for i in range(num_clients):
        client_X = X_train_tensor[client_indices[i]]
        if partition_type=="IID":
            noise_levels = [(i+1)/(num_clients+1) for i in range(num_clients)]#[40*(i/4) for i in range(num_clients)]#[1,10,20]#[1,20,30]#np.linspace(0.1, 10, num_clients)  # Different noise levels for each client
            client_labels = y_train_tensor[client_indices[i]]
            client_y=add_randomized_response_noise(client_labels,noise_levels[i])
        elif partition_type=="N-IID":
            client_y=y_train_tensor[client_indices[i]]

        X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
            client_X, client_y, test_size=local_test_size, stratify=client_y, random_state=42)
    
        train_dataset = TensorDataset(X_train_local, y_train_local)
        test_dataset = TensorDataset(X_test_local, y_test_local)

        client_train_dataloaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True))
        client_test_dataloaders.append(DataLoader(test_dataset, batch_size=32, shuffle=False))
    
    client_data={}
    for client in range(num_clients):
        client_data[client] = {"train_loader": client_train_dataloaders[client], "test_loader": client_test_dataloaders[client]}
    return client_data

def commun_test_set_stroke():
    """
    Creates a DataLoader for the common test set.
    
    Args:
        X_test (torch.Tensor): Test features.
        y_test (torch.Tensor): Test labels.
    
    Returns:
        DataLoader: DataLoader for the common test set.
    """
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

if __name__ == "__main__":
    def_=partition_data_stroke(num_clients=3, alpha=0.5, local_test_size=0.2, partition_type='N-IID')
    common=commun_test_set_stroke()
    print(len(def_[0]["train_loader"].dataset))
    print(len(def_[1]["train_loader"].dataset))
    print(len(def_[2]["train_loader"].dataset))
    print(len(common.dataset))

