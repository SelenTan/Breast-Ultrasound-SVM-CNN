import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import numpy as np

# basic config for dataset location and device choice
DATA_ROOT = "../../Datasets"
DATA_FLAG = "breastmnist"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AddGaussianNoise:
    # tiny transform that adds gaussian noise after ToTensor
    # keeps it as a callable so it plugs into torchvision.Compose
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor is already C,H,W
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class SimpleCNN(nn.Module):
    # small cnn for 28x28 grayscale images
    # width_mult lets us scale channels without changing the structure
    def __init__(self, width_mult=1.0):
        super().__init__()

        # scale channels and avoid zero channels when width is small
        ch1 = max(1, int(16 * width_mult))
        ch2 = max(1, int(32 * width_mult))

        # two conv blocks with pooling to reduce 28x28 down to 7x7
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

        # flatten then classify into 2 classes
        self.fc = nn.Sequential(
            nn.Linear(ch2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x is N,1,28,28
        x = self.layer1(x)
        x = self.layer2(x)

        # flatten per sample, keep batch dimension
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_loader(split, augment_mode="none", max_samples_ratio=1.0):
    # builds the dataset and dataloader for a given split
    # for train split we can add augmentation and optionally subsample data
    info = INFO[DATA_FLAG]
    dataset_class = getattr(medmnist, info["python_class"])

    transform_steps = []

    if split == "train" and augment_mode != "none":
        # rotation expects PIL, so it must happen before ToTensor
        if augment_mode in ["rotation", "combined"]:
            transform_steps.append(transforms.RandomRotation(degrees=15))

        # after this point we are working with tensors
        transform_steps.append(transforms.ToTensor())

        # noise is tensor based, so it goes after ToTensor
        if augment_mode in ["noise", "combined"]:
            transform_steps.append(AddGaussianNoise(mean=0.0, std=0.1))
    else:
        transform_steps.append(transforms.ToTensor())

    # keep scale consistent across runs so training is less jumpy
    transform_steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    final_transform = transforms.Compose(transform_steps)

    dataset = dataset_class(
        split=split,
        transform=final_transform,
        download=True,
        root=DATA_ROOT,
    )

    if split == "train" and max_samples_ratio < 1.0:
        # simulate smaller training budget by sampling a subset of the train set
        sample_count = int(len(dataset) * max_samples_ratio)
        sample_indices = np.random.choice(len(dataset), sample_count, replace=False)
        dataset = data.Subset(dataset, sample_indices)

    loader = data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=(split == "train"),
    )
    return loader


def train_eval_cnn(width_mult=1.0, augment_mode="none", max_samples_ratio=1.0, epochs=15):
    # trains a cnn and returns the best test accuracy seen across epochs
    # also returns the y_true and y_pred from that best epoch
    model = SimpleCNN(width_mult=width_mult).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_loader("train", augment_mode, max_samples_ratio)
    test_loader = get_loader("test", "none")

    best_acc = 0.0
    best_y_true = []
    best_y_pred = []

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE)

            # medmnist labels often come as shape N,1 so squeeze it into N
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.squeeze().long().to(DEVICE)

                logits = model(images)
                predicted = torch.argmax(logits, dim=1)
                
                # Metrics accumulation
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # calculate epoch stats
        epoch_acc = correct / total
        
        # update best results
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_y_true = y_true
            best_y_pred = y_pred
    return best_acc, best_y_true, best_y_pred