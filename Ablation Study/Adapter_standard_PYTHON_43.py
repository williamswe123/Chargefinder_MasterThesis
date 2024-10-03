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
        self.transformer = SimpleTransformer(input_size=hidden_channels, hidden_layer_size=transformer_hidden_size, output_size=out_channels, seq_length=36, num_layers=transformer_num_layers, nhead=transformer_nhead).cuda()

    def forward(self, data, inference=False):
        batch = data.batch
        label = data.y
        label = torch.squeeze(label, 2)
        data.x = data.x.float()  # Convert node features to Double
        data.edge_attr = data.edge_attr.float()  # Convert edge attributes to Double
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.GCN(x, edge_index, edge_attr, batch)
        x = reshape_to_batches(x, batch)
        last_value = reshape_to_batches(data.x[:,-1,:],batch)
        label = reshape_to_batches(label, batch)
        predictions = []
        for station_data, station_label, station_last_value in zip(x.permute(1,0,2,3), label.permute(1,0,2,3), last_value.permute(1,0,2)):
            output = self.transformer(station_data, station_label, station_last_value, inference)
            predictions.append(output)
        predictions = torch.stack(predictions, dim=1)
        return predictions

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1 / math.sqrt(Q.size(-1)))
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)

class AdapterModule(nn.Module):
    def __init__(self, input_size, output_size, bottleneck_size, dropout_rate=0.1):
        super(AdapterModule, self).__init__()
        self.reduce = nn.Linear(input_size, bottleneck_size)
        self.expand = nn.Linear(bottleneck_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_reduced = self.relu(self.reduce(x))
        x_expanded = self.expand(x_reduced)
        return x_expanded

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu
        self.adapter1 = AdapterModule(input_size=d_model, output_size=d_model, bottleneck_size=2)
        self.adapter2 = AdapterModule(input_size=d_model, output_size=d_model, bottleneck_size=2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.adapter1(src2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.adapter2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SimpleTransformer_new(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, nhead, seq_length, num_layers=1, dropout=0.1):
        super(SimpleTransformer_new, self).__init__()
        self.seq_length = seq_length
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.embeddingIn = nn.Linear(input_size, hidden_layer_size)
        self.embeddingTGT = nn.Linear(output_size, hidden_layer_size)
        self.PositionalEncoding = PositionalEncoding(max_len=1000, d_model=hidden_layer_size)
        encoder_layers = CustomTransformerEncoderLayer(d_model=hidden_layer_size, nhead=nhead, dim_feedforward=4*hidden_layer_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_layer_size, nhead=nhead, dim_feedforward=4*hidden_layer_size, dropout=dropout, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_layer_size, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, tgt=None, last_value=None, inference=False):
        last_value = torch.unsqueeze(last_value, dim=2)
        initial_tgt = last_value
        tgt_input = torch.cat([last_value, tgt[:, :-1]], dim=1)
        x = self.embeddingIn(x)
        x = self.PositionalEncoding(x)
        enc_mask = self.generate_square_subsequent_mask(x.size(1)).to(tgt.device)
        x = x.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(x, mask=enc_mask)
        encoder_output = encoder_output.permute(1, 0, 2)
        if inference:
            tgt_gen = initial_tgt
            generated_sequence = torch.zeros((initial_tgt.size(0), self.seq_length, self.output_size), device=x.device)
            encoder_output = encoder_output.permute(1,0,2)
            for i in range(self.seq_length):
                tgt_emb = self.embeddingTGT(tgt_gen)
                tgt_emb = self.PositionalEncoding(tgt_emb)
                tgt_emb = tgt_emb.permute(1, 0, 2)
                decoder_output = self.transformer_decoder(tgt_emb, encoder_output)
                output_step = self.linear1(decoder_output[-1, :, :])
                output_step = output_step.unsqueeze(1) 
                generated_sequence[:, i:i+1, :] = output_step
                tgt_gen = torch.cat((tgt_gen, output_step), dim=1)
                if tgt_gen.size(1) > self.seq_length:
                    tgt_gen = tgt_gen[:, 1:, :]
            return generated_sequence
        else:
            tgt = self.embeddingTGT(tgt_input)
            tgt = self.PositionalEncoding(tgt)
            tgt = tgt.permute(1, 0, 2)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            encoder_output = encoder_output.permute(1,0,2)
            decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)
            output = self.linear1(decoder_output)
            return output.permute(1, 0, 2)

class STGCN_new(nn.Module):
    def __init__(self, in_channels, gcn_layers, hidden_channels, transformer_hidden_size, transformer_num_layers, transformer_nhead, out_channels, bottleneck_size):
        super(STGCN_new, self).__init__()
        self.GCN = GCN(in_channels=in_channels, gcn_hidden_channels=hidden_channels, gcn_layers=gcn_layers)
        self.transformer = SimpleTransformer_new(input_size=hidden_channels, hidden_layer_size=transformer_hidden_size, output_size=out_channels, seq_length=36, num_layers=transformer_num_layers, nhead=transformer_nhead).cuda()

    def forward(self, data, inference=False):
        batch = data.batch
        label = data.y
        label = torch.squeeze(label, 2)
        data.x = data.x.float()  # Convert node features to Double
        data.edge_attr = data.edge_attr.float()  # Convert edge attributes to Double
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.GCN(x, edge_index, edge_attr, batch)
        x = reshape_to_batches(x, batch)
        last_value = reshape_to_batches(data.x[:,-1,:],batch)
        label = reshape_to_batches(label, batch)
        predictions = []
        for station_data, station_label, station_last_value in zip(x.permute(1,0,2,3), label.permute(1,0,2,3), last_value.permute(1,0,2)):
            output = self.transformer(station_data, station_label, station_last_value, inference)
            predictions.append(output)
        predictions = torch.stack(predictions, dim=1)
        return predictions

def transfer_weights(original_model, new_model):
    new_model.embeddingIn.weight.data = original_model.embeddingIn.weight.data.clone()
    new_model.embeddingIn.bias.data = original_model.embeddingIn.bias.data.clone()
    new_model.embeddingTGT.weight.data = original_model.embeddingTGT.weight.data.clone()
    new_model.embeddingTGT.bias.data = original_model.embeddingTGT.bias.data.clone()
    new_model.PositionalEncoding.pe.data = original_model.PositionalEncoding.pe.data.clone()
    for i in range(len(original_model.transformer_encoder.layers)):
        new_model_layer = new_model.transformer_encoder.layers[i]
        original_model_layer = original_model.transformer_encoder.layers[i]
        new_model_layer.self_attn.in_proj_weight.data = original_model_layer.self_attn.in_proj_weight.data.clone()
        new_model_layer.self_attn.in_proj_bias.data = original_model_layer.self_attn.in_proj_bias.data.clone()
        new_model_layer.self_attn.out_proj.weight.data = original_model_layer.self_attn.out_proj.weight.data.clone()
        new_model_layer.self_attn.out_proj.bias.data = original_model_layer.self_attn.out_proj.bias.data.clone()
        new_model_layer.linear1.weight.data = original_model_layer.linear1.weight.data.clone()
        new_model_layer.linear1.bias.data = original_model_layer.linear1.bias.data.clone()
        new_model_layer.linear2.weight.data = original_model_layer.linear2.weight.data.clone()
        new_model_layer.linear2.bias.data = original_model_layer.linear2.bias.data.clone()
        new_model_layer.norm1.weight.data = original_model_layer.norm1.weight.data.clone()
        new_model_layer.norm1.bias.data = original_model_layer.norm1.bias.data.clone()
        new_model_layer.norm2.weight.data = original_model_layer.norm2.weight.data.clone()
        new_model_layer.norm2.bias.data = original_model_layer.norm2.bias.data.clone()
    for i in range(len(original_model.transformer_decoder.layers)):
        new_model_layer = new_model.transformer_decoder.layers[i]
        original_model_layer = original_model.transformer_decoder.layers[i]
        new_model_layer.self_attn.in_proj_weight.data = original_model_layer.self_attn.in_proj_weight.data.clone()
        new_model_layer.self_attn.in_proj_bias.data = original_model_layer.self_attn.in_proj_bias.data.clone()
        new_model_layer.self_attn.out_proj.weight.data = original_model_layer.self_attn.out_proj.weight.data.clone()
        new_model_layer.self_attn.out_proj.bias.data = original_model_layer.self_attn.out_proj.bias.data.clone()
        new_model_layer.multihead_attn.in_proj_weight.data = original_model_layer.multihead_attn.in_proj_weight.data.clone()
        new_model_layer.multihead_attn.in_proj_bias.data = original_model_layer.multihead_attn.in_proj_bias.data.clone()
        new_model_layer.multihead_attn.out_proj.weight.data = original_model_layer.multihead_attn.out_proj.weight.data.clone()
        new_model_layer.multihead_attn.out_proj.bias.data = original_model_layer.multihead_attn.out_proj.bias.data.clone()
        new_model_layer.linear1.weight.data = original_model_layer.linear1.weight.data.clone()
        new_model_layer.linear1.bias.data = original_model_layer.linear1.bias.data.clone()
        new_model_layer.linear2.weight.data = original_model_layer.linear2.weight.data.clone()
        new_model_layer.linear2.bias.data = original_model_layer.linear2.bias.data.clone()
        new_model_layer.norm1.weight.data = original_model_layer.norm1.weight.data.clone()
        new_model_layer.norm1.bias.data = original_model_layer.norm1.bias.data.clone()
        new_model_layer.norm2.weight.data = original_model_layer.norm2.weight.data.clone()
        new_model_layer.norm2.bias.data = original_model_layer.norm2.bias.data.clone()
        new_model_layer.norm3.weight.data = original_model_layer.norm3.weight.data.clone()
        new_model_layer.norm3.bias.data = original_model_layer.norm3.bias.data.clone()
    new_model.linear1.weight.data = original_model.linear1.weight.data.clone()
    new_model.linear1.bias.data = original_model.linear1.bias.data.clone()

def do_da_test(station, random_seed=42, subsample=1, epochs=20, verbose=True):
    future_steps = 36
    seq_len = 576
    batch_size = 64
    warmup_steps = int(epochs * 0.1)
    learning_rate = 0.05
    train_loader, val_loader, test_loader = getLoader(station=station, future_steps=future_steps, seq_len=seq_len, batch_size=batch_size, random_seed=random_seed, subsample=subsample)
    in_channels = 1
    gcn_layers = 3
    hidden_channels = 4
    transformer_hidden_size = 12
    transformer_num_layers = 2
    transformer_nhead = 2
    out_channels = 1
    bottleneck_size = 2
    
    model = STGCN_new(in_channels=in_channels, gcn_layers=gcn_layers, hidden_channels=hidden_channels, transformer_hidden_size=transformer_hidden_size, transformer_num_layers=transformer_num_layers, transformer_nhead=transformer_nhead, out_channels=out_channels, bottleneck_size=bottleneck_size).cuda()
    
    model2 = STGCN_new(in_channels=in_channels, gcn_layers=gcn_layers, hidden_channels=hidden_channels, transformer_hidden_size=transformer_hidden_size, transformer_num_layers=transformer_num_layers, transformer_nhead=transformer_nhead, out_channels=out_channels, bottleneck_size=bottleneck_size).cuda()
    
    original_state_dict = torch.load('Transfer Learning/trained_on_varnamo43.pth')
    
    model.load_state_dict(original_state_dict, strict=False)

    def compare_weights(model1, model2):
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if 'adapter' not in name1:  # Skip adapter layers
                if not torch.equal(param1, param2):
                    print(f"Mismatch found in {name1}")
                else:
                    print(f"Match found in {name1}")

    def transfer_gcn_weights(model1, model2):
        model2.GCN.in_conv.load_state_dict(model1.GCN.in_conv.state_dict())
        for layer1, layer2 in zip(model1.GCN.hidden_convs, model2.GCN.hidden_convs):
            layer2.load_state_dict(layer1.state_dict())

    def compare_gcn_weights(model1, model2):
        for (name1, param1), (name2, param2) in zip(model1.GCN.named_parameters(), model2.GCN.named_parameters()):
            if not torch.equal(param1, param2):
                print(f"Mismatch found in GCN layer {name1}")
            else:
                print(f"Match found in GCN layer {name1}")

    transfer_weights(model.transformer, model2.transformer)
    print("Checking weights after transfer:")
    compare_weights(model.transformer, model2.transformer)
    print("Transferring GCN weights...")
    transfer_gcn_weights(model, model2)
    print("Checking GCN weights after transfer:")
    compare_gcn_weights(model, model2)

    model = model2
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
    if verbose:
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                
        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params

        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {trainable_params / total_params * 100:.2f} %")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    def lr_lambda(current_step: int, d_model: int, warmup_steps: int) -> float:
        current_step += 1
        return (d_model ** (-0.5)) * min((current_step ** (-0.5)), current_step * (warmup_steps ** (-1.5)))
    
    d_model = transformer_hidden_size
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, d_model, warmup_steps))

    best_model, best_epoch, train_losses, val_losses, test_losses, lrs = a_proper_training(
        epochs, model, optimizer, criterion, train_loader, val_loader, test_loader, scheduler, verbose=verbose
    )

    return best_model, train_losses, val_losses, test_losses, best_epoch

if __name__ == "__main__":
    best_model, train_losses, val_losses, best_epoch = do_da_test("varberg")
    torch.save(best_model.state_dict(), "Transfer Learning/trained_on_varnamo-finetuned_on_varberg_DANIEL.pth")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("MSE Loss")
    plt.legend()
    best_model.eval()
    predictAndDisplay(station, val_loader, best_model)
