import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math

futuresteps = 36

def reshape_to_batches(x, batch_description):
    """
        Does something like this:
        torch.Size([28, 576, 64]) --> torch.Size([4, 7, 576, 64])
    """
    num_splits = batch_description.max().item() + 1
    new_shape_dim_0 = num_splits
    new_shape_dim_1 = x.size(0) // new_shape_dim_0
    new_shape = torch.Size([new_shape_dim_0, new_shape_dim_1] + list(x.size()[1:]))
    reshaped_tensor = x.view(new_shape)
    return reshaped_tensor

class GCN(torch.nn.Module):
    def __init__(self, in_channels=1, gcn_hidden_channels=8, gcn_layers=1):
        super(GCN, self).__init__()
        self.in_conv = GCNConv(in_channels, gcn_hidden_channels)
        self.hidden_convs = nn.ModuleList([GCNConv(gcn_hidden_channels, gcn_hidden_channels) for _ in range(gcn_layers - 1)])

    def forward(self, x, edge_index, edge_weight, batch):
        x = x.float()
        x = self.in_conv(x, edge_index, edge_weight)
        for conv in self.hidden_convs:
            x = F.relu(x)
            x = conv(x, edge_index, edge_weight)
        x = F.relu(x)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, nhead, seq_length, num_layers=1, dropout=0.1):
        super(SimpleTransformer, self).__init__()

        self.seq_length = seq_length
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        
        self.embeddingIn = nn.Linear(input_size, hidden_layer_size)
        self.embeddingTGT = nn.Linear(output_size, hidden_layer_size)
        
        self.PositionalEncoding = PositionalEncoding(max_len=1000, d_model=hidden_layer_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=nhead, 
                                                    dim_feedforward=4*hidden_layer_size, dropout=dropout, 
                                                    activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        # tr
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_layer_size, nhead=nhead,
                                                    dim_feedforward=4*hidden_layer_size, dropout=dropout, 
                                                    activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=num_layers)

        self.linear1 = nn.Linear(hidden_layer_size, output_size)
                
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
        
    def forward(self, x, tgt=None, last_value=None, inference=False):
        last_value = torch.unsqueeze(last_value, dim=2)

        initial_tgt = last_value#x[:, -1:]
        #start_value = x[:, -1:]
        
        tgt_input = torch.cat([last_value, tgt[:, :-1]], dim=1)
        
        x = self.embeddingIn(x)
        x = self.PositionalEncoding(x)
        enc_mask = self.generate_square_subsequent_mask(x.size(1)).to(tgt.device)
        x = x.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(x, mask=enc_mask)
        encoder_output = encoder_output.permute(1, 0, 2)
        
        if inference:
            tgt_gen = initial_tgt
            #print(encoder_output.shape)
            #encoder_output = encoder_output.permute(1, 0, 2)
            #print(encoder_output.shape)
            #print(tgt_gen.shape)
            generated_sequence = torch.zeros((initial_tgt.size(0), self.seq_length, self.output_size), device=x.device)
            encoder_output = encoder_output.permute(1,0,2)

            for i in range(self.seq_length):
                #print(tgt_gen.shape)
                
                tgt_emb = self.embeddingTGT(tgt_gen)
                #print(tgt_emb.shape)
                
                tgt_emb = self.PositionalEncoding(tgt_emb)
                tgt_emb = tgt_emb.permute(1, 0, 2)
                #print(tgt_emb.shape)

                decoder_output = self.transformer_decoder(tgt_emb, encoder_output)

                output_step = self.linear1(decoder_output[-1, :, :])
                output_step = output_step.unsqueeze(1) 

                generated_sequence[:, i:i+1, :] = output_step

                tgt_gen = torch.cat((tgt_gen, output_step), dim=1)
                #start_value = torch.unsqueeze(x[:, -1:, 1], 1)

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
            #try dropout here
            output = self.linear1(decoder_output)

            return output.permute(1, 0, 2)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Correct the shaping of pe to [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #print(x.shape)
        #print(self.pe[:, :x.size(1), :].shape)
        # Adjust slicing of pe to match the sequence length of x
        # pe is broadcasted correctly across the batch dimension
        return x + self.pe[:, :x.size(1), :]

class STGCN(nn.Module):
    
    def __init__(self, in_channels, gcn_layers, hidden_channels, transformer_hidden_size, transformer_num_layers, transformer_nhead, out_channels):
        super(STGCN, self).__init__()
        """print("\033[100mhidden_channels:", hidden_channels,
              "   GCN hidden layers:", gcn_layers,
              "   transformer_hidden_size:", transformer_hidden_size,
              "   transformer_num_layers:", transformer_num_layers,
              "   transformer_nhead:", transformer_nhead, "\033[0m")"""

        self.GCN = GCN(in_channels=in_channels, gcn_hidden_channels=hidden_channels, gcn_layers=gcn_layers)

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
