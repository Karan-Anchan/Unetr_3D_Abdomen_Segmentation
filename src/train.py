import os
import torch
from torch.optim import Adam
from monai.losses import DiceLoss
from src.data_loader import get_dataloaders
from src.model import get_model
from src.evaluate import evaluate
from src.utils import save_checkpoint


def train(data_dir, epochs=100, batch_size=2, learning_rate=1e-4, model_dir="models"):
    """Train UNETR with Dice loss, evaluating and checkpointing every 10 epochs."""
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model = get_model()
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)

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
