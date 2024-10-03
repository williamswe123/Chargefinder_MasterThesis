import sys
import os
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
from functions.STGCN import STGCN

def do_da_test(pretrain_station="varnamo", finetune_station="varberg", pretrain_seed=42, finetune_seed=42, subsample=1, epochs = 20, verbose=True):
    
    future_steps = 36
    seq_len = 576
    batch_size = 64
    
    warmup_steps = int(epochs * 0.2)
    learning_rate = 0.0025
    
    
    # Use the function
    train_loader, val_loader, test_loader = getLoader(station=finetune_station, future_steps=future_steps, seq_len=seq_len, batch_size=batch_size, random_seed=finetune_seed, subsample=subsample)
    
    in_channels = 1
    gcn_layers = 3
    hidden_channels = 4
    transformer_hidden_size = 12
    transformer_num_layers = 2
    transformer_nhead = 2
    out_channels = 1
    
    model = STGCN(in_channels=in_channels,
                  gcn_layers=gcn_layers,
                  hidden_channels=hidden_channels,
                  transformer_hidden_size=transformer_hidden_size,
                  transformer_num_layers=transformer_num_layers,
                  transformer_nhead=transformer_nhead,
                  out_channels=out_channels).cuda()
    
    model.load_state_dict(torch.load('Transfer Learning/trained_on_' + pretrain_station + str(pretrain_seed) + '.pth'))
    
    """
    for layer in best_model.GCN.hidden_convs:
        for param in layer.parameters()[:-1]:
            param.requires_grad = False
    """
        
    # Freeze all layers except the last layer of GCN
    for param in model.parameters():
        param.requires_grad = False
        #if "hidden_convs" in name and "[-1]" not in name:
         #   param.requires_grad = False

    # Freeze all layers except the last layer of the Transformer
    for name, param in model.transformer.named_parameters():
        if "transformer_decoder.layers.1" in name:# and "linear1" not in name:
            param.requires_grad = True

    """
    for name, param in model.transformer.named_parameters():
        if param.requires_grad == True:
            print(name, "True")

    for name, param in model.transformer.named_parameters():
        if param.requires_grad == False:
                    print(name, "False")
    """
    if verbose:
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

    best_model, train_losses, val_losses, test_losses, best_epoch = do_da_test(subsample=0.05)

    count_parameters(model)

    torch.save(best_model.state_dict(), "Transfer Learning/trained_on_varnamo-finetuned_on_varberg_wfrieze.pth")
    
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    #plt.plot(lrs, label="learning rates")
    
    plt.title("MSE Loss")
    plt.legend()
    
    best_model.eval()
    
    predictAndDisplay(station, test_loader, best_model)