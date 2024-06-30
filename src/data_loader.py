import os
from glob import glob
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)

def get_dataloaders(data_dir, batch_size, split_ratio=0.8):
    
    """
    Create data loaders for training and validation datasets.

    Args:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Number of samples per batch.
        split_ratio (float): Ratio of data to use for training. Rest will be used for validation.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    
    # Define transformations
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Load data
    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image, "label": label} for image, label in zip(images, labels)]
    train_files, val_files = data_dicts[:int(len(data_dicts) * split_ratio)], data_dicts[int(len(data_dicts) * split_ratio):]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
