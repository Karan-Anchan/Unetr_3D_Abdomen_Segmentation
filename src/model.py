import torch
from monai.networks.nets import UNETR

def get_model():
    
    """
    Create and return a UNETR model.

    Returns:
        torch.nn.Module: UNETR model for 3D segmentation.
    """
    
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(128, 128, 128),
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
