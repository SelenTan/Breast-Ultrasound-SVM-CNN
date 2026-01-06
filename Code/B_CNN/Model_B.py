import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import numpy as np
import random
import os

# basic config shared across experiments
DATA_ROOT = "../../Datasets"
DATA_FLAG = "breastmnist"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    # keep randomness under control so repeated runs are comparable
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AddGaussianNoise:
    # custom tensor level noise transform
    # implemented as a callable so it plugs into torchvision pipelines
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class SimpleCNN(nn.Module):
    # lightweight cnn designed for 28x28 grayscale images
    # width_mult scales channel count to control model capacity
    def __init__(self, width_mult=1.0):
        super().__init__()

        # avoid zero channels when width_mult is small
        ch1 = max(1, int(16 * width_mult))
        ch2 = max(1, int(32 * width_mult))

        # two conv blocks reduce spatial size from 28x28 to 7x7
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # simple fully connected head for binary classification
        self.fc = nn.Sequential(
            nn.Linear(ch2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        # flatten per sample while keeping batch dimension
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_loader(split, augment_mode="none", max_samples_ratio=1.0, batch_size=64, seed=42):
    # build a dataloader with optional augmentation and data subsampling
    # augmentation is only applied to the train split
    set_seed(seed)

    info = INFO[DATA_FLAG]
    dataset_class = getattr(medmnist, info["python_class"])

    # base pipeline used for raw images, validation, and test
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # rotation pipeline runs on PIL before tensor conversion
    rot_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # noise pipeline works on tensors after normalization
    # kept consistent with Model A noise logic
    noise_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        AddGaussianNoise(mean=0.0, std=0.1)
    ])

    # load the base dataset first so we can reuse indices if needed
    base_dataset = dataset_class(
        split=split,
        transform=base_transform,
        download=True,
        root=DATA_ROOT
    )

    # handle training budget by selecting a fixed subset of indices
    indices = np.arange(len(base_dataset))
    if split == "train" and max_samples_ratio < 1.0:
        sample_count = int(len(base_dataset) * max_samples_ratio)
        indices = np.random.choice(len(base_dataset), sample_count, replace=False)
        base_dataset = data.Subset(base_dataset, indices)

    final_dataset = base_dataset

    # build augmented datasets only for training
    if split == "train" and augment_mode != "none":
        datasets_to_concat = [base_dataset]

        # rotation augmentation branch
        if augment_mode in ["rotation", "parallel"]:
            ds_rot = dataset_class(
                split=split,
                transform=rot_transform,
                root=DATA_ROOT
            )
            # reuse the same indices so augmentation matches the same images
            if max_samples_ratio < 1.0:
                ds_rot = data.Subset(ds_rot, indices)
            datasets_to_concat.append(ds_rot)

        # noise augmentation branch
        if augment_mode in ["noise", "parallel"]:
            ds_noise = dataset_class(
                split=split,
                transform=noise_transform,
                root=DATA_ROOT
            )
            if max_samples_ratio < 1.0:
                ds_noise = data.Subset(ds_noise, indices)
            datasets_to_concat.append(ds_noise)

        # combined mode ends up stacking original, rotated, and noisy samples
        final_dataset = data.ConcatDataset(datasets_to_concat)

    shuffle = (split == "train")
    loader = data.DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def evaluate_loader(model, loader):
    # shared evaluation helper used for validation and test
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            logits = model(images)
            predicted = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    return acc, y_true, y_pred


def train_and_validate(width_mult=1.0, augment_mode="none", max_samples_ratio=1.0, epochs=15):
    # train a model and select the checkpoint with best validation accuracy
    set_seed(42)

    model = SimpleCNN(width_mult=width_mult).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_loader("train", augment_mode, max_samples_ratio)
    val_loader = get_loader("val", "none")

    best_val_acc = 0.0
    best_model_state = None

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc, _, _ = evaluate_loader(model, val_loader)

        # keep the best performing model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    return best_val_acc, best_model_state


def run_test_evaluation(width_mult, model_state, augment_mode="none"):
    # reload the selected model and evaluate once on the test set
    model = SimpleCNN(width_mult=width_mult).to(DEVICE)
    model.load_state_dict(model_state)

    test_loader = get_loader("test", "none")
    acc, y_true, y_pred = evaluate_loader(model, test_loader)
    return acc, y_true, y_pred
