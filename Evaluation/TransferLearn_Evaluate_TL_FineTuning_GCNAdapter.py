import sys
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


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
from functions.display_things import *
from functions.trainFuncs import *
from functions.STGCN import *

def do_da_test(pretrain_station="varnamo", finetune_station="varberg", pretrain_seed=42, finetune_seed=42, subsample=1, epochs = 20, verbose=True):
    
    
    future_steps = 36
    seq_len = 576
    batch_size = 64
    
    warmup_steps = int(epochs * 0.2)
    learning_rate = 0.005
    
    # Use the function
    train_loader, val_loader, test_loader = getLoader(station=finetune_station, future_steps=future_steps, seq_len=seq_len, batch_size=batch_size, random_seed=finetune_seed, subsample=subsample)
    
    in_channels = 1
    gcn_layers = 3
    hidden_channels = 4
    transformer_hidden_size = 12
    transformer_num_layers = 2
    transformer_nhead = 2
    out_channels = 1
    
    class STGCNnew(nn.Module):
        
        def __init__(self, in_channels, gcn_layers, hidden_channels, transformer_hidden_size, transformer_num_layers, transformer_nhead, out_channels):
            super(STGCNnew, self).__init__()
            
            self.GCN = GCN(in_channels=in_channels, gcn_hidden_channels=hidden_channels, gcn_layers=gcn_layers)
    
            self.transformer = SimpleTransformer(input_size = hidden_channels, hidden_layer_size=transformer_hidden_size,
                                                 output_size=out_channels, seq_length=36, num_layers=transformer_num_layers,
                                                 nhead=transformer_nhead).cuda()
                    
            # Adapter layers
            self.AdapterLayerBeg1 = GCNConv(1, 1)
            
            self.AdapterLayerMiddle1 = GCNConv(hidden_channels, 1)
            self.AdapterLayerMiddle2 = GCNConv(1, hidden_channels)
    
            
        def forward(self, data, inference=False):    
            batch = data.batch
            label = data.y
            label = torch.squeeze(label, 2)
            
            data.x = data.x.float()  # Convert node features to Double
            data.edge_attr = data.edge_attr.float()  # Convert edge attributes to Double
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
           
            x = self.AdapterLayerBeg1(x, edge_index, edge_attr)
        
            # Spatial processing
            x = self.GCN(x, edge_index, edge_attr, batch)
            
            x = self.AdapterLayerMiddle1(x, edge_index)
            x = F.relu(x)
            x = self.AdapterLayerMiddle2(x, edge_index)
            
            x = reshape_to_batches(x, batch)
            last_value = reshape_to_batches(data.x[:,-1,:],batch)
            label = reshape_to_batches(label, batch)
                    
            # Reshape and pass data through the model for each station
            predictions = []
           
            for station_data, station_label, station_last_value in zip(x.permute(1,0,2,3), label.permute(1,0,2,3), last_value.permute(1,0,2)):
                output = self.transformer(station_data, station_label, station_last_value, inference)
                predictions.append(output)
    
            # Concatenate predictions for all stations
            predictions = torch.stack(predictions, dim=1)
            return predictions
    
    # Example usage:
    # Define the adjacency matrix for spatial processing (A_spatial)
    # Define the input size, number of layers, and number of heads for the temporal transformer
    original_state_dict = torch.load('Transfer Learning/trained_on_' + pretrain_station + str(pretrain_seed) + '.pth')
    
    model = STGCNnew(in_channels=1, gcn_layers=3, hidden_channels=4, transformer_hidden_size=12,
                      transformer_num_layers=2, transformer_nhead=2, out_channels=1).cuda()
    
    new_state_dict = model.state_dict()
    for name, param in original_state_dict.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            new_state_dict[name] = param
    
    model.load_state_dict(new_state_dict)
    
    for name, param in model.named_parameters():
        if name in original_state_dict.keys():
            param.requires_grad = False

    if verbose:
        print()
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name, param.requires_grad)
        print()
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print(name, param.requires_grad)

        count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()    
    
    # Define the lambda function for scheduling with Noam-style learning rate decay
    def lr_lambda(current_step: int, d_model: int, warmup_steps: int) -> float:
        current_step+=1
        return (d_model ** (-0.5)) * min((current_step ** (-0.5)), current_step * (warmup_steps ** (-1.5)))
    
    d_model = transformer_hidden_size
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, d_model, warmup_steps))    
    
    best_model, best_epoch, train_losses, val_losses, test_losses, lrs = a_proper_training(
        epochs, model, optimizer, criterion, train_loader, val_loader, test_loader, scheduler, verbose=verbose
    )
    
    return best_model, train_losses, val_losses, test_losses, best_epoch

if __name__ == "__main__":
    
    best_model, train_losses, val_losses, test_losses, best_epoch = do_da_test(subsample=0.5)
    
    torch.save(best_model.state_dict(), "Transfer Learning/trained_on_varnamo-finetuned_on_varberg_GCNAdapter.pth")
    
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    #plt.plot(lrs, label="learning rates")
    
    plt.title("MSE Loss")
    plt.legend()
    
    best_model.eval()
    
    predictAndDisplay(station, test_loader, best_model)
