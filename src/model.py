import torch
from monai.networks.nets import UNETR

def create_model():
    
    """
    Create and return a UNETR model instance.
    
    Returns:
        UNETR: A UNETR model instance.
    """
    
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    return model
