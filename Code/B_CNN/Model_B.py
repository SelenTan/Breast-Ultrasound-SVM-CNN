import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

DATA_ROOT = '../../Datasets'
DATA_FLAG = 'breastmnist'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    # Added width_mult parameter to control Capacity
    def __init__(self, width_mult=1.0):
        super(SimpleCNN, self).__init__()
        
        # Base filters scaled by multiplier
        f1 = int(16 * width_mult)
        f2 = int(32 * width_mult)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(f2 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 2) 
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_loader(split, augment_mode):
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    transform_list = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    
    if augment_mode and split == 'train':
        aug_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]
        final_transforms = transforms.Compose(aug_list + transform_list)
    else:
        final_transforms = transforms.Compose(transform_list)

    dataset = DataClass(split=split, transform=final_transforms, download=False, root=DATA_ROOT)
    return data.DataLoader(dataset, batch_size=64, shuffle=(split=='train'))

# Updated to accept capacity parameter
def train_cnn_model(augment_mode, epochs=15, capacity_mult=1.0):
    print(f"CNN Training: Augmentation={augment_mode}, Capacity={capacity_mult}")
    train_loader = get_loader('train', augment_mode)
    test_loader = get_loader('test', False)

    model = SimpleCNN(width_mult=capacity_mult).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        history.append(correct / total)
        # print(f"Epoch {epoch+1} Acc: {history[-1]:.4f}") # Optional reduce clutter
    
    return history

def run_best_model():
    print("Running Best CNN Model")
    history = train_cnn_model(augment_mode=True, epochs=15, capacity_mult=1.0)
    print(f"Final Accuracy: {history[-1]:.4f}")

if __name__ == '__main__':
    run_best_model()