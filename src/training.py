from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import Adam

def train(model, train_loader, epochs, device):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")
