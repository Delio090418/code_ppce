import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import math
from modelos import MLP,CNN,CNN_brain,LogisticRegressionModel
#import sys
#sys.path.append('/Users/delio/Documents/Working projects/Balazs/Experiments/MNIST')
#from data_partition import data_for_client, commun_test_set
from tqdm import tqdm
from torchvision.transforms import ToTensor
import random
from local_breast import partition_data,commun_test_set
import torch.optim as optim
from data_partition_local import data_for_clients#,commun_test_set
from data_stroke import partition_data_stroke, commun_test_set_stroke
from data_partition_noise_labels import data_for_clients_brain, commun_test_set_brain


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

class Client:
    def __init__(self, model_type, dataset, device=dev):
        self.device = device
        self.data_loader = dataset["train_loader"]
        self.test=dataset["test_loader"]
        self.model_type=model_type
        #self.testing_cli=dataset[1]
        self.model = self.init_model(model_type).to(device)
        self.criterion, self.optimizer=self.opt_and_cri(model_type)
        # self.criterion = self.op_cr[0]
        # self.optimizer = self.op_cr[1]

        self.client_size = len(self.data_loader.dataset) + len(self.test.dataset)


    def opt_and_cri(self,model_type):
        cr1=nn.CrossEntropyLoss()#nn.functional.cross_entropy
        # cr2=nn.BCELoss()
        op1=torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.001)#optim.SGD(self.model.parameters(), lr=0.01)
        # op2=optim.SGD(self.model.parameters(), lr=0.01)
        if model_type == "MLP":
            return cr1, op1 
        elif model_type == "CNN":
            return cr1, op1
        elif model_type == "CNN_brain" or "LOG_S":
            return cr1, op1
        elif model_type == "LOG_B" :
            return nn.BCELoss(), optim.SGD(self.model.parameters(), lr=0.01)
        else:
            raise ValueError("Unsupported Criterion and Optimizer")
        
   
    def init_model(self, model_type):
        if model_type == "MLP":
            return MLP()
        elif model_type == "CNN":
            return CNN()
        elif model_type == "CNN_brain":
            return CNN_brain()
        elif model_type == "LOG_B":
            return LogisticRegressionModel(input_dim=30)
        elif model_type == "LOG_S":    
            return LogisticRegressionModel(input_dim=10)
        else:
            raise ValueError("Unsupported model type")

    
    def fit(self, num_epochs):#fit(self, num_epochs test,plot=False):
        # Training loop
        #lenght=len(self.data_loader)
        self.model.to(dev)
        #batch_losses = []  # Store batch-wise loss
        #epoch_losses = [] 
        #val_losses = []
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            #epoch_loss = 0  # Accumulate epoch loss
            #num_batches = len(self.data_loader)
            for data, target in self.data_loader:
                data, target = data.to(dev), target.to(dev)
                self.optimizer.zero_grad()
                if self.model_type == "LOG_B" or "LOG_S":
                    output = self.model(data).squeeze(dim=1)
                else:
                    output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # batch_losses.append(loss.item())  # Store batch loss
                # epoch_loss += loss.item()
            # epoch_losses.append(epoch_loss / num_batches)
            # # Evaluate on Validation Set
            # self.model.eval()  # Set model to evaluation mode
            # epoch_val_loss = 0

            # with torch.no_grad():
            #     for inputs, targets in test:
            #         inputs, targets = inputs.to(dev), targets.to(dev)
            #         outputs = self.model(inputs).squeeze(dim=1)
            #         loss_val = self.criterion(outputs, targets)
            #         epoch_val_loss += loss_val.item()

            # val_losses.append(epoch_val_loss / len(test))  # Store average validation loss

        # if plot==True:
        #     plt.figure(figsize=(8, 4))
        #     plt.plot(range(1, num_epochs + 1), epoch_losses, label="Training Loss", color="blue")
        #     plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color="red")            
        #     plt.xlabel("Epochs")
        #     plt.ylabel("Loss")
        #     plt.title("Epoch-wise Training Loss")
        #     plt.legend()
        #     plt.show()

    
    def evaluation(self,test_set):
        self.model.eval()
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for data, target in test_set:
                data, target = data.to(dev), target.to(dev)
                if self.model_type == "LOG_B" or self.model_type =="LOG_S":
                    outputs = self.model(data).squeeze(dim=1)
                    predicted = (outputs >= 0.5).float()
                else: 
                    outputs = self.model(data)
                    _, predicted = torch.max(outputs, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:
                    break
        return correct/total
    
    def get_model_grads(self):
        return [param.grad for param in self.model.parameters()]
    
    
    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def parameters(self):
        return self.model.parameters()

class federation:
    def __init__(self, model_type, dataset_name, num_clients, partition_type='N-IID',alpha=0.5,device=dev):
        self.alpha=alpha
        self.num_clients=num_clients
        self.model_type=model_type
        self.dataset_name=dataset_name
        self.partition_type=partition_type
        #self.sizes=client_sizes
        self.model = self.init_model(model_type).to(device)
        self.datos=self.load_dataset(self.dataset_name,self.num_clients)
        self.datasets=self.datos[0]
        self.test_global=self.datos[1]
        self.lista_clientes=[Client(model_type,self.datasets[i]) for i in range(self.num_clients)]



    def load_dataset(self, dataset_name, n_cli):
        if dataset_name == "MNIST":
            data_mnist=data_for_clients(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test_mnist=commun_test_set(data_mnist)
            return data_mnist, test_mnist
        elif dataset_name == "CIFAR10":
            data_cifar=data_for_clients(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test_cifar=commun_test_set(data_cifar)
            return data_cifar, test_cifar
        elif dataset_name=="BRAIN":
            data_brain =data_for_clients_brain(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test_brain = commun_test_set_brain()
            return data_brain, test_brain
        elif dataset_name=="BREAST":
            data=partition_data()#(num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test=commun_test_set(data)
            return data, test
        elif dataset_name=="STROKE":
            data=partition_data_stroke(num_clients=n_cli, alpha=self.alpha, partition_type=self.partition_type)#(num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test=commun_test_set_stroke()
            return data, test
        else:
            raise ValueError("Unsupported dataset")
        


    def init_model(self, model_type):
        if model_type == "MLP":
            return MLP()
        elif model_type == "CNN":
            return CNN()
        elif model_type == "CNN_brain":
            return CNN_brain()
        elif model_type == "LOG_B":
            return LogisticRegressionModel(input_dim=30)
        elif model_type == "LOG_S":    
            return LogisticRegressionModel(input_dim=10)
        else:
            raise ValueError("Unsupported model type")


    def total_size(self):
        size = 0
        for client in self.lista_clientes:
            size += client.client_size
        return size


    def aggregate(self, List_clients):
        pesos_cliente = [client.parameters() for client in List_clients]
        data = []
        for params_clients in zip(*pesos_cliente):
            data.append(sum(param.data for param in params_clients) / len(List_clients))
        return data
    
    def weighted_aggre(self,List_clients):
        pesos_cliente = [client.parameters() for client in List_clients] 
        total_data = sum(cliente.client_size for cliente in List_clients)
        data_amount=[participant.client_size for participant in List_clients]
        data = []
        for params_clients in zip(*pesos_cliente):
            order_para=[]
            for param, amout in zip(params_clients,tuple(data_amount)):
                order_para.append((amout/total_data)*param.data)
            data.append(sum(order_para))
        return data
    
    def federated_averaging(self, num_iterations, conjunto_cli,num_epochs=20):
        conj_clientes = [self.lista_clientes[i] for i in conjunto_cli]
        iteration = 0
        while iteration <= num_iterations - 1:
            for client in conj_clientes:
                weights_modelo = self.model.state_dict()
                client.load_state_dict(weights_modelo)
                client.fit(num_epochs)
            tqdm.write(f"fed iteration {iteration}")
            params = self.weighted_aggre(conj_clientes)
            self.set_parameters(params)
            iteration += 1
        #return self.evaluacion(test_set)  
        
    def evaluation_global(self,test_set):
        self.model.eval()
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for data, target in test_set:
                data, target = data.to(dev), target.to(dev)
                if self.model_type== "LOG_B" or self.model_type=="LOG_S":
                    outputs = self.model(data).squeeze(dim=1)
                    predicted = (outputs >= 0.5).float()
                else: 
                    outputs = self.model(data)
                    _, predicted = torch.max(outputs, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:
                    break
        return correct/total


    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def parametros(self):
        data=[param.data for param in self.model.parameters()]
        return data


if __name__ == "__main__":
    #data=partition_data_stroke(num_clients=3, alpha=7.0, local_test_size=0.2, partition_type='N-IID')
    #data_for_clients("BRAIN",3,0.5,partition_type="IID")
    #data=dirichlet_partition(3, 0.5)
    # cli=Client("LOGISTIC",data[0])
    # cli.fit(1)
    # eval=cli.evaluation(cli.test)
    # print(eval)
    fed=federation("LOG_S","STROKE",3)
    clients=fed.lista_clientes
    fed.federated_averaging(10,[0,1,2],3)
    eva_1=fed.evaluation_global(clients[1].test)
    eva_2=fed.evaluation_global(fed.test_global)
    print(eva_1)
    print(eva_2)




    # mlp = Client("MLP","MNIST",0)
    # mlp.fit(10)
    # test_set=mlp.testing_cli
    # eva=mlp.evaluation(test_set)
    # print(eva)