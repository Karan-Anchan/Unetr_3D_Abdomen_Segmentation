from monai.losses import DiceCELoss
from torch.optim import Adam

def train(model, train_loader, epochs, device):
    
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on.
    """
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = Adam(model.parameters(), 1e-4)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
