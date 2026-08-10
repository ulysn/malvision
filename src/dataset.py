from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(augment=False):
    steps = [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if augment:
        steps += [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            transforms.RandomRotation(degrees=5),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


class MalwareDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_manifest(processed_dir):
    processed_dir = Path(processed_dir)
    family_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    class_names = [d.name for d in family_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    records = []
    for family_dir in family_dirs:
        label = class_to_idx[family_dir.name]
        for img_path in sorted(family_dir.glob('*.png')):
            records.append({'path': str(img_path), 'family': family_dir.name, 'label': label})

    return pd.DataFrame(records), class_names, class_to_idx


def create_splits(processed_dir, data_dir='data', random_state=42):
    df, class_names, _ = build_manifest(processed_dir)
    paths, labels = df['path'].tolist(), df['label'].tolist()

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=random_state
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.50, stratify=temp_l, random_state=random_state
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    pd.DataFrame({'path': train_p, 'label': train_l}).to_csv(data_dir / 'train.csv', index=False)
    pd.DataFrame({'path': val_p,   'label': val_l  }).to_csv(data_dir / 'val.csv',   index=False)
    pd.DataFrame({'path': test_p,  'label': test_l }).to_csv(data_dir / 'test.csv',  index=False)
    pd.DataFrame({'idx': range(len(class_names)), 'family': class_names}).to_csv(
        data_dir / 'classes.csv', index=False
    )

    print(f"Splits — train: {len(train_p)}  val: {len(val_p)}  test: {len(test_p)}")
    return train_p, train_l, val_p, val_l, test_p, test_l, class_names


def load_splits_from_csv(data_dir='data'):
    data_dir = Path(data_dir)
    train_df  = pd.read_csv(data_dir / 'train.csv')
    val_df    = pd.read_csv(data_dir / 'val.csv')
    test_df   = pd.read_csv(data_dir / 'test.csv')
    classes   = pd.read_csv(data_dir / 'classes.csv')['family'].tolist()

    return (
        train_df['path'].tolist(), train_df['label'].tolist(),
        val_df['path'].tolist(),   val_df['label'].tolist(),
        test_df['path'].tolist(),  test_df['label'].tolist(),
        classes,
    )


def get_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(train_paths, train_labels, val_paths, val_labels,
                    test_paths, test_labels, batch_size=64, num_workers=4):
    train_ds = MalwareDataset(train_paths, train_labels, transform=get_transforms(augment=True))
    val_ds   = MalwareDataset(val_paths,   val_labels,   transform=get_transforms(augment=False))
    test_ds  = MalwareDataset(test_paths,  test_labels,  transform=get_transforms(augment=False))

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent)

    return train_loader, val_loader, test_loader
