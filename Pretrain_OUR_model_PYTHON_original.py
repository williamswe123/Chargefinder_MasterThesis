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
from functions.trainFuncs import a_proper_training

from functions.STGCN import *

class STGCN(nn.Module):
    
    def __init__(self, in_channels, gcn_layers, hidden_channels, transformer_hidden_size, transformer_num_layers, transformer_nhead, out_channels, bottleneck_size):
        super(STGCN, self).__init__()

        self.GCN = GCN(in_channels=in_channels, gcn_hidden_channels=hidden_channels, gcn_layers=gcn_layers)
        
        self.adapter = AdapterModule(input_size=hidden_channels, output_size=hidden_channels, bottleneck_size=bottleneck_size)
        
        self.transformer = SimpleTransformer(input_size = hidden_channels, hidden_layer_size=transformer_hidden_size,
                                             output_size=out_channels, seq_length=36, num_layers=transformer_num_layers,
                                             nhead=transformer_nhead).cuda()
        
    def forward(self, data, inference=False):    
        batch = data.batch
        label = data.y
        label = torch.squeeze(label, 2)
        
        data.x = data.x.float()  # Convert node features to Double
        data.edge_attr = data.edge_attr.float()  # Convert edge attributes to Double
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
       
        # Spatial processing
        x = self.GCN(x, edge_index, edge_attr, batch)
        
        x = self.adapter(x)
        
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

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.scale / math.sqrt(Q.size(-1)))
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)


class AdapterModule(nn.Module):
    def __init__(self, input_size, output_size, bottleneck_size, dropout_rate=0.1):
        super(AdapterModule, self).__init__()
        self.reduce = nn.Linear(input_size, bottleneck_size)
        self.attention = CrossModalAttention(bottleneck_size)
        self.expand = nn.Linear(bottleneck_size, output_size)
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x_reduced = self.relu(self.reduce(x))
        gating_signal = torch.sigmoid(self.gate(x))
        attention_mask = gating_signal.squeeze(-1) > 0.5
        attention_mask = attention_mask.unsqueeze(-1).expand_as(x_reduced)
        x_attention = torch.where(attention_mask, self.attention(x_reduced), x_reduced)
        x_attention = self.dropout(x_attention)
        x_expanded = self.expand(x_attention)

        # Apply residual connection
        if x.size(-1) == x_expanded.size(-1):
            x_expanded += x  # Residual connection
        x_expanded = self.norm(x_expanded)  # Apply normalization

        return self.scale * x_expanded
    
def do_da_test(station, random_seed = 42, subsample=1, epochs = 20, verbose=True):
    
    future_steps = 36
    seq_len = 576
    batch_size = 64
    
    warmup_steps = int(epochs * 0.2)
    learning_rate = 0.05
    
    # Use the function
    train_loader, val_loader, test_loader = getLoader(station=station, future_steps=future_steps,
                                                      seq_len=seq_len, batch_size=batch_size,
                                                      random_seed=random_seed, subsample=subsample)
    
    in_channels = 1
    gcn_layers = 3
    hidden_channels = 4
    transformer_hidden_size = 12
    transformer_num_layers = 2
    transformer_nhead = 2
    out_channels = 1
    bottleneck_size = 2
    
    model = STGCN(in_channels=in_channels,
                  gcn_layers=gcn_layers,
                  hidden_channels=hidden_channels,
                  transformer_hidden_size=transformer_hidden_size,
                  transformer_num_layers=transformer_num_layers,
                  transformer_nhead=transformer_nhead,
                  out_channels=out_channels,
                  bottleneck_size=bottleneck_size).cuda()
    
    #count_parameters(model)

    # Freeze all parameters
    #for param in model.parameters():
    #    param.requires_grad = False

    # Unfreeze the adapter parameters
    #for name, param in model.named_parameters():
    #    if 'adapter' in name:
    #        param.requires_grad = True

    # Optionally, check which parameters are trainable
    if verbose:
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)  # This should only print the names of the adapter parameters

        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params

        # Assuming 'best_model' is your model instance
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {trainable_params / total_params * 100:.2f} %")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()    
    
    # Define the lambda function for scheduling with Noam-style learning rate decay
    def lr_lambda(current_step: int, d_model: int, warmup_steps: int) -> float:
        current_step+=1
        return (d_model ** (-0.5)) * min((current_step ** (-0.5)), current_step * (warmup_steps ** (-1.5)))
    
    d_model = transformer_hidden_size
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, d_model, warmup_steps))    
    
    best_model, best_epoch, train_losses, val_losses, test_losses, lrs = a_proper_training(
        epochs, model, optimizer, criterion, train_loader, val_loader, test_loader, scheduler, verbose=True
    )
    
    return best_model, train_losses, val_losses, test_losses, best_epoch

if __name__ == "__main__":

    best_model, train_losses, val_losses, best_epoch = do_da_test("varberg")
    
    torch.save(best_model.state_dict(), "Transfer Learning/trained_on_varnamo-finetuned_on_varberg_DANIEL.pth")
    
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    #plt.plot(lrs, label="learning rates")
    
    plt.title("MSE Loss")
    plt.legend()
    
    best_model.eval()
    
    predictAndDisplay(station, val_loader, best_model)