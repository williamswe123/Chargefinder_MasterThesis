import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
import time
import copy

import random

import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
import math

from torch.utils.data import Dataset, DataLoader

# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class datasetMaker(Dataset):
    def __init__(self, data, indices_conversion, seq_len=10, future_steps=5):
        # Assuming 'data' is a numpy array or a pandas DataFrame, convert it to a numpy array
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.indices_conversion = indices_conversion
        self.seq_len = seq_len
        self.future_steps = future_steps

    def __len__(self):
        # Subtract seq_len to avoid going out of bounds
        return len(self.indices_conversion)

    def __getitem__(self, index):
        # Get the sequence and label, and convert them to torch tensors
        index = self.indices_conversion[index]
        #random_index = random.randint(0,len(self.data)-self.seq_len-1)
        #random_index = 1000
        seq = torch.tensor(self.data[index:index+self.seq_len], dtype=torch.float)
        label = torch.tensor(self.data[index+self.seq_len:index+self.seq_len+self.future_steps], dtype=torch.float)
        label=torch.unsqueeze(label[:,0], 1)
        
        return seq, label
    
def train_epoch(epoch, optimizer, loss_function, model, train_loader, future_steps):
    total_loss = 0
    model.train()
    for batch_idx, (data,label) in enumerate(train_loader):

        data = data.cuda()
        label = label.cuda()
                
        optimizer.zero_grad()
        
        predictions = model(data, future=future_steps)
                            
        loss_value = loss_function(predictions,label)
        loss_value.backward()
        optimizer.step()
        
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def validate_epoch(epoch, loss, model, val_loader, future_steps):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            
            data = data.cuda()
            label = label.cuda()
            
            predictions = model(data, future=future_steps)
            
            
            loss_value = loss(predictions, label)
            total_loss += loss_value.item()
    return total_loss / len(val_loader)

def a_proper_training(num_epoch, model, optimizer, loss_function, train_loader, val_loader, future_steps, verbose):
    best_epoch = None
    best_model = None
    best_loss = None
    train_losses = list()
    val_losses = list()

    if verbose:
        print("Begin Training")

    for epoch in range(num_epoch):
        start_time = time.time()  # Start time

        train_loss = train_epoch(epoch, optimizer, loss_function, model, train_loader, future_steps)
        val_loss = validate_epoch(epoch, loss_function, model, val_loader, future_steps)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch == 0:
            best_loss = val_loss
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epoch}: Train Loss = {train_loss} Val Loss = {val_loss} Elapsed_time = {elapsed_time}")
            
    return (best_model, best_epoch, train_losses, val_losses)



class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MultiStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Flatten LSTM parameters
        self.lstm.flatten_parameters()
        
        # Fully connected layer to map LSTM output to desired output_size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, future=1):
        
        predictions = []
        
        
        for _ in range(future):
            # Initialize hidden state and cell state        
            batch_size, sequence_length, _ = x.size()
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

            # LSTM forward pass
            out, (h0, c0) = self.lstm(x, (h0, c0))
            
            pred = out[:, -1, :]
            
            x = torch.cat((x, pred.unsqueeze(1)), dim=1)
            
            t = self.fc(pred)
            predictions.append(t) # Append occupancy to predictions
            

        # Stack predictions along the sequence length dimension'
        predictions = torch.cat(predictions, dim=1)
        
        predictions = torch.unsqueeze(predictions, dim = 2)
        return predictions

def do_da_test(station, epochs, verbose, seed):
    if verbose:
        print()

    set_seed(seed)
    
    data = pd.read_csv('../data/' + station)
    # Adding / removing columns.
    
    # Loading the data
    data.set_index('Unnamed: 0', inplace=True)
    data = data.drop(columns=data.columns.difference(['Occupancy']))
    
    future_steps = 36
    seq_len = 576
    batch_size = 8
    
    indices = list(range(100, len(data) - future_steps - seq_len - 300))

    random.shuffle(indices)

    indices = indices[:]
    
    train_i = indices[:int(len(indices)*0.8)] 
    val_i = indices[int(len(indices)*0.8):int(len(indices)*0.9)]
    test_i = indices[int(len(indices)*0.9):]

    
    train_dataset = datasetMaker(data, train_i, seq_len, future_steps)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
    val_dataset = datasetMaker(data, val_i, seq_len, future_steps)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_dataset = datasetMaker(data, test_i, seq_len, future_steps)
    test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = MultiStepLSTM(input_size=1, hidden_size=1, output_size=1, num_layers=3).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_function = nn.MSELoss()
    
    best_model, best_epoch, train_losses, val_losses = a_proper_training(
        epochs, model, optimizer, loss_function, train_loader, val_loader, future_steps, verbose
    )


    return best_model, train_losses, val_losses, best_epoch


if __name__ == "__main__":
    do_da_test("varnamo/data_IONITY_LOCF.csv", 1, True, 42)
