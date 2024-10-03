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

def reshape_from_batches(x):
    """
        Reshapes a tensor from batched shape back to original shape.
        torch.Size([4, 7, 576, 64]) --> torch.Size([28, 576, 64])
    """
    # Get the new dimensions
    num_splits, new_shape_dim_1 = x.size(0), x.size(1)
    
    # Compute the original shape
    original_shape_dim_0 = num_splits * new_shape_dim_1
    original_shape = torch.Size([original_shape_dim_0] + list(x.size()[2:]))
    
    # Reshape the tensor to the original shape
    reshaped_tensor = x.view(original_shape)
    return reshaped_tensor


def train_epoch(epoch, optimizer, loss_function, model, train_loader):
    total_loss = 0
    model.train()
    for batch_idx, (data,label) in enumerate(train_loader):

        label = reshape_to_batches(label, data.batch)
        data = data.cuda()
        label = label.cuda().float()
                
        optimizer.zero_grad()
        predictions = model(data, inference=False)
                
        loss_value = loss_function(predictions,label)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def validate_epoch(epoch, loss, model, val_loader):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):

            label = reshape_to_batches(label, data.batch)
            data = data.cuda()
            label = label.cuda().float()
            
            predictions = model(data, inference=True)
            
            loss_value = loss(predictions, label)
            total_loss += loss_value.item()
    return total_loss / len(val_loader)
import time
import copy

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

        if lstm:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
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
        
    return (best_model, best_epoch, train_losses, val_losses, test_losses, lrs)
