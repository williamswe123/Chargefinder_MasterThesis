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
from functions.STGCN import STGCN

station = "varberg"
future_steps = 36
seq_len = 576
batch_size = 64
random_seed = 42

epochs = 20
warmup_steps = int(epochs * 0.2)
learning_rate = 0.005


# Use the function
train_loader, val_loader, test_loader = getLoader(station=station, future_steps=future_steps,
                                                  seq_len=seq_len, batch_size=batch_size,
                                                  random_seed=random_seed)

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


class Adapter(nn.Module):
    def __init__(self, input_size, hidden_network):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, input_size)
        self.hidden_network = hidden_network
        self.relu = nn.ReLU()

    def forward(self, data, inference):
        x = data.x
        x = reshape_to_batches(x, data.batch)
        batch_size, stations, seq_len, features = x.shape
        
        x = x.view(batch_size, -1)        
        # Apply the fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # Reshape back to original shape
        x = x.view(64, 5, 576, 1)        
        
        x = reshape_from_batches(x)
        data.x = x
        
        x = self.hidden_network(data, inference)
        return x
    
model.load_state_dict(torch.load('Transfer Learning/trained_on_varnamo42.pth'))

adapter_network = Adapter(2880, model).cuda()

for param in adapter_network.hidden_network.parameters():
    param.requires_grad = False
for param in adapter_network.fc1.parameters():
    param.requires_grad = True
for param in adapter_network.fc2.parameters():
    param.requires_grad = True

count_parameters(model)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()    

# Define the lambda function for scheduling with Noam-style learning rate decay
def lr_lambda(current_step: int, d_model: int, warmup_steps: int) -> float:
    current_step+=1
    return (d_model ** (-0.5)) * min((current_step ** (-0.5)), current_step * (warmup_steps ** (-1.5)))

d_model = transformer_hidden_size
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, d_model, warmup_steps))    

best_model, best_epoch, train_losses, val_losses, lrs = a_proper_training(
    epochs, adapter_network, optimizer, criterion, train_loader, val_loader, scheduler
)

torch.save(best_model.state_dict(), "trained_on_varnamo-finetuned_on_varberg_Adapter.pth")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
#plt.plot(lrs, label="learning rates")

plt.title("MSE Loss")
plt.legend()


best_model.eval()

predictAndDisplay(station, test_loader, best_model)