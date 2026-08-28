import os
import numpy as np
import rasterio
from torch.utils.data import Dataset


class OpenEarthMapDataset(Dataset):
    """PyTorch Dataset for OpenEarthMap flat layout.

    Directory layout expected:
        ROOT_DIR/
        ├── <img_dir>/*.tif   (e.g. images/train, images/val)
        └── <mask_dir>/*.tif  (e.g. labels/train, labels/val — some mirrors use "label/")

    `split_file=None` (the baseline default): NO splitting/subsetting at all —
    every image under <img_dir> that has a matching mask under <mask_dir> is
    used directly, i.e. exactly the train/val partition the dataset already
    ships with.

    `split_file=<path>` (optional, kept only for future use — e.g. a 5-fold
    spatial CV list): restricts samples to the basenames listed in that file,
    one per line (path prefixes such as "tyrolw/tyrolw_25" are accepted —
    os.path.basename strips them down to "tyrolw_25").
    """

    CLASSES = [
        'Background', 'Bareland', 'Rangeland', 'Developed',
        'Road', 'Tree', 'Water', 'Agriculture', 'Building',
    ]
    NUM_CLASSES = 9
    IGNORE_INDEX = 255

    def __init__(self, root_dir, img_dir, mask_dir, split_file=None, transform=None):
        self.root_dir  = root_dir
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        img_full_dir  = os.path.join(root_dir, img_dir)
        mask_full_dir = os.path.join(root_dir, mask_dir)

        if split_file is None:
            # No split file -> use every image/mask pair already present in
            # the dataset's own train/val directory, as-is.
            assert os.path.isdir(img_full_dir), f"Image directory not found: {img_full_dir}"
            names = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(img_full_dir)
                if f.lower().endswith('.tif')
            )
        else:
            assert os.path.exists(split_file), f"Split file not found: {split_file}"
            with open(split_file) as f:
                names = [os.path.basename(line.strip()) for line in f if line.strip()]

        self.samples = []
        for name in names:
            if not name:
                continue
            img_path  = os.path.join(img_full_dir,  name + '.tif')
            mask_path = os.path.join(mask_full_dir, name + '.tif')
            assert os.path.exists(img_path),  f"Image not found: {img_path}"
            assert os.path.exists(mask_path), f"Mask not found:  {mask_path}"
            self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        with rasterio.open(img_path) as src:
            image = src.read()                        # (C, H, W) uint8
        image = np.transpose(image, (1, 2, 0))        # (H, W, C)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        image = image.astype(np.uint8)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)       # (H, W)

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result['image']                   # float tensor (C, H, W)
            mask  = result['mask'].long()             # int64 tensor (H, W)

        return image, mask
