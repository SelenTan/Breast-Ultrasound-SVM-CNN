import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import numpy as np
import random

DATA_ROOT = "../../Datasets"
DATA_FLAG = "breastmnist"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    # set seeds so repeated runs are comparable
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

        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_loader(split, augment_mode="none", max_samples_ratio=1.0, batch_size=64, seed=42):
    # build a dataloader with optional augmentation and data subsampling
    # augmentation is only applied to the train split
    set_seed(seed)
    info = INFO[DATA_FLAG]
    dataset_class = getattr(medmnist, info["python_class"])

    # base pipeline used for raw images, validation, and test
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    # Augmentations:
    if split == "train":
        if augment_mode == "rotation":
            transform_list.insert(0, transforms.RandomRotation(degrees=15))
            
        elif augment_mode == "noise": 
            transform_list.append(AddGaussianNoise(mean=0.0, std=0.1))
            
        elif augment_mode == "parallel":
            transform_list.insert(0, transforms.RandomRotation(degrees=15))
            transform_list.append(AddGaussianNoise(mean=0.0, std=0.1))

    final_transform = transforms.Compose(transform_list)

    dataset = dataset_class(
        # load the base dataset first so we can reuse indices if needed
        split=split,
        transform=final_transform,
        download=True,
        root=DATA_ROOT
    )

    # training budget
    if split == "train" and max_samples_ratio < 1.0:
        sample_count = int(len(dataset) * max_samples_ratio)
        indices = np.random.choice(len(dataset), sample_count, replace=False)
        dataset = data.Subset(dataset, indices)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loader


def evaluate_loader(model, loader):
    # shared evaluation helper used for validation and test
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze(-1).long().to(DEVICE)

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
        for images, labels in train_loader: #for every batch
            images = images.to(DEVICE)
            labels = labels.squeeze(-1).long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc, _, _ = evaluate_loader(model, val_loader)

        # note the best performing model 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    return best_val_acc, best_model_state


def run_test_evaluation(width_mult, model_state):
    # reload the selected model and evaluate once on the test set
    model = SimpleCNN(width_mult=width_mult).to(DEVICE)
    model.load_state_dict(model_state)

    test_loader = get_loader("test", "none")
    acc, y_true, y_pred = evaluate_loader(model, test_loader)
    return acc, y_true, y_pred