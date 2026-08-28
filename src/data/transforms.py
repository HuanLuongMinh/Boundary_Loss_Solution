import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_MEAN = [123.675, 116.28, 103.53]
_STD  = [58.395,  57.12,  57.375]


def get_train_transforms(crop_size: int = 512, resize: int = 1024):
    return A.Compose([
        A.Resize(height=resize, width=resize),
        A.RandomCrop(height=crop_size, width=crop_size),
        # Geometric — applied to both image and mask
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Color — image only (mask unchanged by albumentations default)
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.Normalize(mean=[m / 255.0 for m in _MEAN], std=[s / 255.0 for s in _STD]),
        ToTensorV2(),
    ])


def get_val_transforms(resize: int = 1024):
    return A.Compose([
        A.Resize(height=resize, width=resize),
        A.Normalize(mean=[m / 255.0 for m in _MEAN], std=[s / 255.0 for s in _STD]),
        ToTensorV2(),
    ])
