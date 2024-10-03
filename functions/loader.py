import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data, Dataset
import networkx as nx

import os
from .graph import getGraph

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
    def __init__(self, station_data, indices_conversion, edge_index, edge_attr, seq_len, future_steps, batch_size, subsample=1):
        self.station_data = station_data
        self.indices_conversion = indices_conversion
        self.size = next(iter(station_data.values())).shape[0]
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.batch_size = batch_size
        self.subsample = subsample

    def __len__(self):
        return len(self.indices_conversion)
    
    def __getitem__(self, index):
        
        index = self.indices_conversion[index]
        
        seq_end = index + self.seq_len
        fut_end = index + self.seq_len + self.future_steps
        
        node_features = []
        for i, (station, data) in enumerate(self.station_data.items()):
            node_feature = data.iloc[index:seq_end].values
            node_features.append(node_feature)
        node_features = torch.tensor(np.array(node_features)).float()

        labels = []
        for i, (station, data) in enumerate(self.station_data.items()):
            label = data.iloc[seq_end:fut_end].values
            labels.append(label)
        labels = torch.unsqueeze(torch.tensor(np.array(labels)), dim=2).float()
        
        Gdata = Data(x=node_features, y=labels, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return Gdata, labels

    
def custom_collate(batch):
    label = torch.cat([i[1] for i in batch])
    label = label.squeeze(3)
    batch = Batch.from_data_list([b[0] for b in batch])
    return batch, label

def getLoader(station=None, future_steps=36, seq_len=576, batch_size=64, subsample=1, random_seed=42):

    set_seed(random_seed)
    
    if station == "varnamo":
        files = ["donken",
                 "Holmgrens",
                 "IONITY",
                 "Jureskogs_Vattenfall",
                 "UFC"]
    elif station == "varberg":
        files = ["Varberg_UFC",
                 "IONITY_Varberg",
                 "Toveks_Bil",
                 "Varberg_Supercharger",
                 "Recharge_ST1"]
    elif station == "malmo":
        files = ["OKQ8_Svansjögatan",
                 "Emporia_plan_3",
                 "P_Huset_Plan_2",
                 "OKQ8_Kvartettgatan",
                 "P_hus_Malmömässan"]
    else:
        print(station, "not yet implimented or spelled wrong")
    
    num_workers=4
    
    data_dict = {}
    
    for f in files:
        if station == 'malmo':
            path = 'data/'+ station +'/data_' + f + '_5T_k-3.csv'
        else:
            path = 'data/'+ station +'/data_' + f + '_5T_k-10.csv'

        # Loading the data
        data = pd.read_csv(path)
        data.set_index('Unnamed: 0', inplace=True)
        data = data.drop(columns=data.columns.difference(['Occupancy']))
        data_dict[f] = data

    # Making indicies
    indices = list(range(len(data) - future_steps - seq_len))

    # Making the test indicies
    test_i = random.sample(indices, int(len(indices)*0.1))
    indices = [i for i in indices if i not in test_i]
    random.shuffle(test_i)

    # Subsampling the data
    if subsample < 1:
        subsampled_data_length = int((len(indices) * subsample))
        start_index = random.randint(0, len(indices) - subsampled_data_length)
        indices = indices[start_index:start_index + subsampled_data_length]

    random.shuffle(indices)

    train_i = indices[:int(len(indices)*0.8)] 
    val_i = indices[int(len(indices)*0.8):]

    edge_index, edge_attr = getGraph(station)
    
    train_dataset = datasetMaker(data_dict, train_i, edge_index, edge_attr, seq_len, future_steps, batch_size, subsample)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate, num_workers=num_workers)
    
    val_dataset = datasetMaker(data_dict, val_i, edge_index, edge_attr, seq_len, future_steps, batch_size, subsample)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate, num_workers=num_workers)
    
    test_dataset = datasetMaker(data_dict, test_i, edge_index, edge_attr, seq_len, future_steps, batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__=="__main__":
    train_loader, val_loader, test_loader = getLoader(station="varnamo", future_steps=36, seq_len=576, batch_size=64, random_seed=42, subsample=0.05)
    print()
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

