import os
import medmnist
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define paths
DATA_ROOT = '../../Datasets'
DATA_FLAG = 'breastmnist'

def get_data_loader(split, augment=False):
    # Standard transformation
    transform_list = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    
    # Add augmentation if requested
    if augment and split == 'train':
        # SVM is sensitive to rotation but we can add small perturbations
        transform_list.insert(0, transforms.RandomHorizontalFlip())
    
    data_transform = transforms.Compose(transform_list)

    info = medmnist.INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])

    # download is False because main.py handles it
    dataset = DataClass(split=split, transform=data_transform, download=False, root=DATA_ROOT)
    
    loader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data_batch, labels = next(iter(loader))
    
    X = data_batch.view(data_batch.size(0), -1).numpy()
    y = labels.numpy().ravel()
    
    return X, y

def train_svm_model(c_value, augment=False):
    """
    Reusable function to train SVM with specific parameters
    Used by both main execution and experiment analysis
    """
    X_train, y_train = get_data_loader('train', augment)
    X_test, y_test = get_data_loader('test', False)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = svm.SVC(kernel='rbf', C=c_value, gamma='scale')
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return acc, clf, y_pred, y_test

def run_best_model():
    print("Running Model A SVM Best Configuration")
    # C 1.0 is selected as the optimal hyperparameter
    acc, _, y_pred, y_test = train_svm_model(c_value=1.0, augment=False)
    
    print(f"Model A SVM Test Accuracy {acc:.4f}")
    print("Classification Report")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

if __name__ == '__main__':
    run_best_model()