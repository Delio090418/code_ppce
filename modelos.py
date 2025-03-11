import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#I use this model for CIFAR10 with num_classes=10, for brain data set with num_classes=4
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def set_parameters(self,params):
        for model_parametro, param in zip(self.parameters(), params):
            model_parametro.data = param
    
#Use this for brain data set
class CNN_brain(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_brain, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x   
    
    def set_parameters(self,params):
        for model_parametro, param in zip(self.parameters(), params):
            model_parametro.data = param


        
   
    

#I use this model MNIST data set
class MLP(nn.Module):
    def __init__(self, neurons=[784,16,10]):  #'self' is required
        super(MLP, self).__init__()  # <- This is required

        # Define all layers as a special list
        self.layers = nn.Sequential()
        #self.normalizer = nn.BatchNorm1d(neurons[0], affine=False, track_running_stats=False)
        #self.normalizer = nn.LayerNorm(neurons[0])
        #self.layers.append(normalizer)
        # This way, we just iteratively append all layers
        for i in range(len(neurons)-2):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
            self.layers.append(nn.ReLU())
        
        # Unfortunately, the last layer has a different activation function
        #(an 'if' statement could alse be introduced in the above loop)
        self.layers.append(nn.Linear(neurons[i+1], neurons[i+2]))
        self.layers.append(nn.Softmax(dim=1))
       
        # self.criterion = nn.CrossEntropyLoss()
        
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self, x):
       # x = self.normalizer(x)
        x = x.view(-1, 28*28)
        x = self.layers(x)
        return x
    
    def set_parameters(self,params):
        for model_parametro, param in zip(self.parameters(), params):
            model_parametro.data = param
    

#This model is for the breast cancer data set 
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=30):
        super(LogisticRegressionModel, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 1)
        #self.linear_2 = nn.Linear(2, 1)
        
    def forward(self, x):
        #x=self.linear_1(x)
        return torch.sigmoid(self.linear_1(x))
    
    def set_parameters(self,params):
        for model_parametro, param in zip(self.parameters(), params):
            model_parametro.data = param


