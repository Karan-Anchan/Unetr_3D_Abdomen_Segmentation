import os
import torch
from torch.optim import Adam
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from src.data_loader import get_dataloaders
from src.model import get_model
from src.utils import save_checkpoint, load_checkpoint

def train(data_dir, epochs=100, batch_size=2, learning_rate=1e-4, model_dir="models"):
    
    """
    Train the UNETR model.

    Args:
        data_dir (str): Path to the directory containing the data.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.
        model_dir (str): Directory to save model checkpoints.
    """
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model = get_model()
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Training Loss: {epoch_loss / len(train_loader)}")

        if (epoch + 1) % 10 == 0:
            val_dice = evaluate(model, val_loader)
            print(f"Validation Dice: {val_dice:.4f}")
            save_checkpoint(model, os.path.join(model_dir, f"model_epoch_{epoch + 1}.pth"))
