from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import torch

def evaluate(model, data_loader, device):
    
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device to run the evaluation on.
    """
    
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    with torch.no_grad():
        for batch_data in data_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            dice_metric(y_pred=outputs, y=labels)
        
        metric = dice_metric.aggregate().item()
        print(f"Evaluation Dice Score: {metric:.4f}")
    
    dice_metric.reset()
