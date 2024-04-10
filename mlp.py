# import libraries
import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.optim import Adam 
import pandas as pd


### ------ Prepare the data ---------

class Data():
    def __init__(self, name_dataset, batch_size):
        self.name_dataset_ = name_dataset
        self.batch_size_ = batch_size

    def prepare_dataset(self):
        print('Creation of the dataset ....')
        df = pd.read_csv(self.name_dataset_, sep=',')
        df.drop(labels=['Date', '4046', '4225', '4770', 'type', 'year', 'region', 'XLarge Bags'], axis=1, inplace=True)
        X = df.loc[:, df.columns != 'AveragePrice']
        Y = df.loc[:,'AveragePrice']

        # Convert Input and Output data to Tensors and create a TensorDataset 
        X = torch.Tensor(X.to_numpy(dtype='float32'))      # Create tensor of type torch.float32 
        
        Y = torch.tensor(Y.to_numpy(dtype='float32')).unsqueeze(1)  # Create tensor type torch.float32 of size (n,1)  
       

        data = TensorDataset(X, Y)
        train_set, test_set = random_split(data, [0.8, 0.2]) 

        # Create Dataloader to read the data within batch sizes and put into memory. 
        train_loader = DataLoader(train_set, batch_size = self.batch_size_, shuffle = True) 
        test_loader = DataLoader(test_set, batch_size = 1)

        print("Train et test datasets created !")
        return train_loader, test_loader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, learning_rate, loss):
        super().__init__()

        self.input_size_ = input_size
        self.output_size_ = output_size
        self.learning_rate_ = learning_rate
        self.loss_ = loss 
        self.mlp_ = nn.Sequential(
            #nn.Flatten(),
            nn.BatchNorm1d(num_features=self.input_size_),
            nn.Linear(self.input_size_, 20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,self.output_size_)
        )
        self.optimizer_ = Adam(self.mlp_.parameters(), lr=self.learning_rate_)

    def forward(self, x):
        value = self.mlp_(x)
        return value
    
    def train(self, num_epoch, training_set):
        #best_accuracy = 0.0
        print("Training is beginning...")

        for epoch in range(1,num_epoch+1):
            for data in training_set:
                X,y = data 
                self.optimizer_.zero_grad()   # zero the parameter gradients          
                predicted_outputs = self.forward(X)   # predict output from the model 
                train_loss = self.loss_(predicted_outputs, y)   # calculate loss for the predicted output  
                train_loss.backward()   # backpropagate the loss 
                self.optimizer_.step()
            print(f"Epoch {epoch} : Loss : {train_loss}")

        print("Training is finished !")

if __name__ == '__main__':
    data = Data(name_dataset='avocado_full.csv',
                batch_size=64)
    
    train, test = data.prepare_dataset()

    model = NeuralNetwork(input_size=4, 
                          output_size=1, 
                          learning_rate=1e-3, 
                          loss=nn.MSELoss())
    model.train(5,train)
