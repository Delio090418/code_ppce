import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from torchvision import models

import torch.optim as optim
from tqdm import tqdm

from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
SHUFFLE = True


image_dir = '/Users/delio/Documents/Working_projects/ISIC_2019/ISIC_2019_Training_Input'
csv_path = "/Users/delio/Documents/Working_projects/ISIC_2019/ISIC_2019_Training_GroundTruth.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


database = pd.read_csv(csv_path)

# Keep only images with single class
database = database[database.iloc[:, 1:].sum(axis=1) == 1].copy()

# Convert one-hot to single label
database['label'] = database.iloc[:, 1:].idxmax(axis=1)

# Convert label to numeric class index
database['label_id'] = database['label'].astype('category').cat.codes


def non_iid_dirichlet_split(df, n_clients=6, alpha=0.5, seed=42):
    np.random.seed(seed)
    label_list = df['label_id'].values
    client_indices = defaultdict(list)
    n_classes = len(np.unique(label_list))
    
    for cls in range(n_classes):
        cls_indices = df[df['label_id'] == cls].index.tolist()
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions / proportions.sum() * len(cls_indices)).astype(int)

        # Handle rounding issues
        while proportions.sum() > len(cls_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(cls_indices):
            proportions[np.argmin(proportions)] += 1

        idx_splits = np.split(np.array(cls_indices), np.cumsum(proportions)[:-1])
        for client_id, split in enumerate(idx_splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


class ISIC2019Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'] + '.jpg')  # this uses image_dir!
        img = Image.open(img_path).convert('RGB')
        label = row['label_id']
        if self.transform:
            img = self.transform(img)
        return img, label

#dataset = ISIC2019Dataset(df, image_dir=image_dir)

def split_client_data(df, test_size=0.2, seed=42):
    label_counts = df['label_id'].value_counts()

    if (label_counts < 2).any():
        # Not enough samples per class for stratification
        print("Warning: Skipping stratification due to low sample counts.")
        return train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    else:
        return train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label_id'])

def non_iid_data(n_clients=6, alpha=0.5, seed=42):
    client_indices = non_iid_dirichlet_split(database, n_clients=n_clients, alpha=alpha, seed=seed)
    client_dfs = [database.loc[indices].copy() for indices in client_indices.values()] 
    client_train_dfs = []
    client_test_dfs = []
    for d in client_dfs:
        train_df, test_df = split_client_data(d)
        client_train_dfs.append(train_df)
        client_test_dfs.append(test_df)

    clients_train_data = [ISIC2019Dataset(f, image_dir, transform=transform) for f in client_train_dfs]

    clients_test_data = [ISIC2019Dataset(f, image_dir, transform=transform) for f in client_test_dfs]

    client_loaders = {}
    
    for i in range(n_clients):
        train_loader = DataLoader(clients_train_data[i], batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        test_loader = DataLoader(clients_test_data[i], batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        client_loaders[i] = {"train_loader": train_loader, "test_loader": test_loader}
    return client_loaders


def commun_test_isic(commun):
        common=[data_loader["test_loader"].dataset  for data_loader in  commun.values()]
        combined=ConcatDataset(common)
        combined_dataloader = DataLoader(combined, batch_size=32, shuffle=True)
        return combined_dataloader

# for i, dataset in enumerate(clients_data):
#     print(f"Client {i}: {len(dataset)} samples")#


# client_loaders = [
#     DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
#     for dataset in clients_data
# ]



# client_loaders = {
#     f"client_{i}": DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
#     for i, dataset in enumerate(clients_data)
# # }

# # print(f"{len(client_loaders["client_0"])}")



# def get_model(num_classes=8):  # ISIC 2019 has 8 classes
#     model = models.resnet50(pretrained=True)  # Load pretrained ResNet50

#     # Replace the final fully connected layer
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)

#     return model

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = get_model().to(device)



# def train_model(model, train_loader, num_epochs=10, lr=1e-4, device='mps'):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     model.to(device)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct, total = 0, 0

#         loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

#         for images, labels in loop:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Stats
#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = 100. * correct / total
#         print(f"Epoch {epoch+1} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#         # # Optional: Validation
#         # if val_loader:
#         #     evaluate_model(model, val_loader, device)

#     return model

# model = get_model(num_classes=8)

# # For one client's loader (you can call this inside a federated loop)
# trained_model = train_model(model, client_loaders["client_0"], num_epochs=1)


