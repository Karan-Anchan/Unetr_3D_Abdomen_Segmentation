import torch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

def evaluate(model, val_loader):
    """Run sliding-window inference over the validation set and return the mean Dice score."""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].cuda(), val_data["label"].cuda()
            val_outputs = sliding_window_inference(val_images, (128, 128, 128), 4, model)
            dice_metric(y_pred=val_outputs, y=val_labels)

    return dice_metric.aggregate().item()
