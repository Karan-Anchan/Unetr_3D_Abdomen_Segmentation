import os
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd,
    RandCropByPosNegLabeld, RandShiftIntensityd, Compose
)
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist

def get_data_loaders(data_dir, batch_size, num_workers):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0)),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])
    
    data_json = os.path.join(data_dir, 'dataset_0.json')
    train_files = load_decathlon_datalist(data_json, is_train=True, data_list_key="training")
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader
