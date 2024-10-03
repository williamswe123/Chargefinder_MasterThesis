import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

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
    

def predictAndDisplay(station, loader, model):
    model.eval()
        
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
        print(station, "not yet implimented")

    for data, label in loader:
        
        label = reshape_to_batches(label, data.batch)
    
        label = label.float().cpu()
        data = data.cuda()
    
        # Get predictions from the model
        predictions = model(data, inference=True)
        
        # Convert tensors to numpy arrays
        predictions = predictions.detach().cpu().numpy()
        label = label.numpy()
        
        # Determine the number of rows and columns for the grid
        batch_size, nr_of_nodes, seq_len, _ = predictions.shape
        num_rows = 4 #batch_size
        num_cols = nr_of_nodes

    
        
            
        # Create a grid of  
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        # Plot each sequence in the batch
        for i in range(4):
            for j, station in enumerate(files):
                
                if j == 5:
                    break
                
                ax = axes[i, j]
    
                # Get the predictions and true values for the current sequence
                t = label[i, j, :, 0]
                p = predictions[i, j, :, 0]
    
                # Plot the predictions and true values on the current subplot
                ax.plot(p, label="Predictions")
                ax.plot(t, label="True Values")
                ax.set_title(f"{station} Batch {i+1}")
                #ax.legend()
                ax.set_ylim(0, 1)
    
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        break  # Break after processing the first batch

    # Assuming dataloader is your DataLoader and model is your neural network model
    
    # Set the model to evaluation mode
    #best_model.cpu()
    
    # Initialize variables to accumulate predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Iterate through all batches in the DataLoader
    with torch.no_grad():
    
        for batch_data, batch_labels in loader:
            batch_labels = reshape_to_batches(batch_labels, batch_data.batch)
    
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda().float()
            # Forward pass: compute predicted outputs by passing inputs to the model
            batch_predictions = model(batch_data, inference=True)
    
            # Append batch predictions and labels to the accumulated lists
            all_predictions.append(batch_predictions)
            all_labels.append(batch_labels)
    
    # Concatenate the lists of predictions and labels into tensors
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    
    
    all_predictions_by_station = torch.chunk(all_predictions, chunks=7, dim=1)
    all_labels_by_station = torch.chunk(all_labels, chunks=7, dim=1)
    
    avg_loss = 0
    
    for p, l, f in zip(all_predictions_by_station, all_labels_by_station, files):
        mse = F.mse_loss(p, l)
        avg_loss += mse
        print("Mean Squared Error for", f, ":\t", mse.item())
    print("\navg loss:", avg_loss.item()/5)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #return total_params, trainable_params
    
    #Assuming 'best_model' is your model instance
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {trainable_params / total_params * 100:.2f} %")
        
