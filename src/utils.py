import os
import torch

def save_checkpoint(model, path):
    
    """
    Save model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the model checkpoint.
    """
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_checkpoint(model, path):
    
    """
    Load model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
