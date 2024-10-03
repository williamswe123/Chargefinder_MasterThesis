import sys
import os
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR

# Get the directory containing the notebook
notebook_dir = os.path.dirname(os.path.abspath("__file__"))

# Add the directory containing the notebook to sys.path
sys.path.append(notebook_dir)

# Add the parent directory (which contains the 'dataloaders' directory) to sys.path
parent_dir = os.path.abspath(os.path.join(notebook_dir, '.'))
sys.path.append(parent_dir)

from functions.loader import getLoader
from functions.trainFuncs import *
from functions.display_things import *
from functions.STGCN import *

future_steps = 36

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
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
    
class STGCN(nn.Module):
    
    def __init__(self, in_channels, gcn_layers, hidden_channels, lstm_layers, out_channels):
        super(STGCN, self).__init__()
        
        self.GCN = GCN(in_channels=in_channels, gcn_hidden_channels=hidden_channels, gcn_layers=gcn_layers)

        self.lstm = MultiStepLSTM(hidden_channels, hidden_channels, out_channels, lstm_layers)
        
    def forward(self, data, inference):    
 
        batch = data.batch
        
        data.x = data.x.float()  # Convert node features to Double
        data.edge_attr = data.edge_attr.float()  # Convert edge attributes to Double
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
       
        # Spatial processing
        x = self.GCN(x, edge_index, edge_attr, batch)

        x = reshape_to_batches(x, batch)
        # Reshape and pass data through the model for each station
        predictions = []
        for station_data in x.permute(1,0,2,3):  # Iterate over each station
            #station_data = station_data.permute(1, 0, 2)  # Reshape for LSTM (batch_first=True)
            output = self.lstm(station_data, future=future_steps)
            predictions.append(output)

        # Concatenate predictions for all stations
        predictions = torch.stack(predictions, dim=1)
        return predictions


def a_proper_training(num_epoch, model, optimizer, loss_function, train_loader, val_loader, test_loader, scheduler, patience=None, hyperN_start=None, verbose=True, lstm=False):
    best_epoch = None
    best_model = None
    best_loss = float('inf')
    train_losses = list()
    val_losses = list()
    test_losses = list()
    lrs = list()

    # Early stopping variables
    patience_counter = 0  # to count the number of epochs without improvement
    stop_training = False
    if verbose:
        print("Begin Training")

    for epoch in range(num_epoch):
        if stop_training:
            break
        
        if (not hyperN_start == None)  and epoch == hyperN_start:
            model.hyper_on(True)
            
        start_time = time.time()  # Start time
        val_loss = validate_epoch(epoch, loss_function, model, val_loader)
        test_loss = validate_epoch(epoch, loss_function, model, test_loader)
        train_loss = train_epoch(epoch, optimizer, loss_function, model, train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        scheduler.step(val_loss)
        
        lrs.append(optimizer.param_groups[0]['lr'])
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            patience_counter = 0  # reset counter if there's an improvement
        else:
            if not patience == None:
                patience_counter += 1  # increment counter if no improvement
        # Patient check
        if not patience == None and patience_counter >= patience:
            if verbose:
                print(f"Stopping early at epoch {epoch + 1}")
            stop_training = True
        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch + 1}/{round(num_epoch, 10)}: Train Loss={round(train_loss,10)} Val Loss={round(val_loss,10)} Test Loss={round(test_loss,10)} Elapsed_time = {round(elapsed_time/60,2)}minutes")
        
    return best_model, best_epoch, train_losses, val_losses, test_losses, lrs



def train_STGCN(station="varnamo", random_seed=42, subsample=1, epochs=20, patience=10,  verbose=True):
    
    future_steps = 36
    seq_len = 576
    batch_size = 64
    
    warmup_steps = int(epochs * 0.2)
    learning_rate = 0.005
    
    
    # Use the function
    train_loader, val_loader, test_loader = getLoader(station=station, future_steps=future_steps, seq_len=seq_len, batch_size=batch_size, random_seed=random_seed, subsample=subsample)
    if verbose:
        print(len(train_loader))
        print(len(val_loader))
        print(len(test_loader))

    model = STGCN(in_channels=1, gcn_layers=1, hidden_channels=8, lstm_layers=1, out_channels=1).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # Now pass the scheduler to the training function
    best_model, best_epoch, train_losses, val_losses, test_losses, lrs = a_proper_training(
        epochs, model, optimizer, criterion, train_loader, val_loader, test_loader, scheduler, verbose=verbose, patience=10
    )
    return best_model, train_losses, val_losses, test_losses, best_epoch

if __name__ == "__main__":
    train_STGCN()
