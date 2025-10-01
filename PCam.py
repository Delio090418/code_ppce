import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd

# === CONFIG ===
BATCH_SIZE = 32
SHUFFLE = True


x_path = '/Users/delio/Documents/Working_projects/camelyonpatch/camelyonpatch_level_2_split_train_x.h5'
y_path = '/Users/delio/Documents/Working_projects/camelyonpatch/camelyonpatch_level_2_split_train_y.h5'

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),  # PCam patches are 96x96
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



class PCamDataset(Dataset):
    def __init__(self, x_data, y_data, indices, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.x_data[real_idx]  # shape: (96, 96, 3)
        label = self.y_data[real_idx]

        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    

def non_iid_dirichlet_split_pcam(y_data, n_clients=6, alpha=0.5, seed=42):
    np.random.seed(seed)
    label_list = np.array(y_data[:])
    client_indices = defaultdict(list)
    n_classes = len(np.unique(label_list))

    for cls in range(n_classes):
        cls_indices = np.where(label_list == cls)[0]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions / proportions.sum() * len(cls_indices)).astype(int)

        # Handle rounding issues
        while proportions.sum() > len(cls_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(cls_indices):
            proportions[np.argmin(proportions)] += 1

        idx_splits = np.split(cls_indices, np.cumsum(proportions)[:-1])
        for client_id, split in enumerate(idx_splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices   

def split_indices_train_test(indices, y_data, test_size=0.2, seed=42):
    labels = y_data[indices]
    if len(np.unique(labels)) < 2:
        # Avoid stratification if only one class
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    else:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    return train_idx, test_idx 



def non_iid_data_pcam(x_path, y_path, n_clients=6, alpha=0.5, seed=42):
    x_data = h5py.File(x_path, 'r')['x']
    y_data = h5py.File(y_path, 'r')['y']

    client_indices = non_iid_dirichlet_split_pcam(y_data, n_clients=n_clients, alpha=alpha, seed=seed)

    client_loaders = {}

    for client_id, indices in client_indices.items():
        train_idx, test_idx = split_indices_train_test(indices, y_data, seed=seed)

        train_dataset = PCamDataset(x_data, y_data, train_idx, transform=transform)
        test_dataset = PCamDataset(x_data, y_data, test_idx, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True)

        client_loaders[client_id] = {
            'train_loader': train_loader,
            'test_loader': test_loader
        }

    return client_loaders


def common_test_pcam(client_loaders):
    test_datasets = [loader["test_loader"].dataset for loader in client_loaders.values()]
    combined_dataset = ConcatDataset(test_datasets)
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True)
    return combined_loader
   


clients = non_iid_data_pcam(x_path, y_path, n_clients=6, alpha=0.3)

global_test_loader = common_test_pcam(clients)

# Example: Train a model per client
for cid, loaders in clients.items():
    print(f"Client {cid}: {len(loaders['train_loader'].dataset)} train, {len(loaders['test_loader'].dataset)} test")