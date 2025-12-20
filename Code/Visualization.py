import torch
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys

# Add path to import your models
sys.path.append('./A_SVM')
sys.path.append('./B_CNN')

# Configure paths
DATA_ROOT = '../Datasets'
DATA_FLAG = 'breastmnist'

def get_data_loader(split):
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DataClass(split=split, transform=data_transform, download=True, root=DATA_ROOT)
    # Get all data
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return loader

def plot_confusion_matrix(y_true, y_pred, model_name):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    # Annotate cells with numbers and use blue color map
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save file
    filename = f'confusion_matrix_{model_name}.png'
    plt.savefig(filename)
    print(f"Saved {filename}")

# Note You need to copy your model logic here or import it
# For simplicity let us generate the matrix for Model B CNN as an example
# You can adapt this for SVM easily

def generate_cnn_matrix():
    print("Generating Confusion Matrix for CNN")
    from B_CNN.Model_B import SimpleCNN, device
    
    # Load Test Data
    loader = get_data_loader('test')
    
    # Initialize Model
    # ideally you load trained weights but here we retrain quickly or just use initialized weights
    # For the report you should save your trained model weights using torch.save
    # But for now let's just show how to plot
    model = SimpleCNN().to(device)
    
    # We assume model is trained. If not results will be random.
    # In a real scenario load weights here: model.load_state_dict(torch.load('path_to_weights.pth'))
    
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(targets.numpy().flatten())
            all_preds.extend(predicted.cpu().numpy().flatten())
            
    plot_confusion_matrix(all_targets, all_preds, "CNN")

if __name__ == '__main__':
    # Ensure you are running this from the Code directory
    generate_cnn_matrix()