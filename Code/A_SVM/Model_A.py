import medmnist
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import scipy.ndimage

DATA_FLAG = 'breastmnist'
DATA_ROOT = '../../Datasets'

def get_data_loader(split, augment_mode='none', max_samples_ratio=1.0, random_seed=42):
    # ensure reproducibility by setting random seed
    np.random.seed(random_seed)
    
    info = medmnist.INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])

    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    dataset = DataClass(
        split=split,
        transform=transforms.Compose(base_transforms),
        download=True,
        root=DATA_ROOT
    )

    # reduce dataset size to simulate limited training budget
    if split == 'train' and max_samples_ratio < 1.0:
        n_samples = int(len(dataset) * max_samples_ratio)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data_batch, labels = next(iter(loader))

    X = data_batch.numpy().squeeze(1)
    y = labels.numpy().ravel()

    # expand training set by concatenating original and augmented samples
    # svm requires explicit data doubling to learn invariant features
    if split == 'train' and augment_mode != 'none':
        X_aug = np.copy(X)
        y_aug = np.copy(y)
        
        if augment_mode == 'rotation':
            for i in range(len(X_aug)):
                angle = np.random.uniform(-15, 15) 
                X_aug[i] = scipy.ndimage.rotate(X_aug[i], angle, reshape=False, mode='nearest')

        elif augment_mode == 'noise':
            noise = np.random.normal(loc=0.0, scale=0.1, size=X_aug.shape)
            X_aug = X_aug + noise

        elif augment_mode == 'parallel':
            # apply composite augmentation involving both rotation and noise
            for i in range(len(X_aug)):
                angle = np.random.uniform(-15, 15) 
                X_aug[i] = scipy.ndimage.rotate(X_aug[i], angle, reshape=False, mode='nearest')
            noise = np.random.normal(loc=0.0, scale=0.1, size=X_aug.shape)
            X_aug = X_aug + noise
            
        # concatenate original and augmented data along the sample axis
        X = np.concatenate((X, X_aug), axis=0)
        y = np.concatenate((y, y_aug), axis=0)

    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, y

def extract_features(X_flat, feature_type='raw'):
    if feature_type == 'raw':
        return X_flat

    if feature_type == 'hog':
        # calculate image dimensions dynamically to avoid fixed size dependencies
        n_features = X_flat.shape[1]
        side_len = int(np.sqrt(n_features))
        
        # restore 2d spatial structure for gradient calculation
        X_imgs = X_flat.reshape(-1, side_len, side_len)
        hog_feats = []
        for i in range(len(X_imgs)):
            f = hog(
                X_imgs[i],
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                visualize=False
                # feature_vector default is True
            )
            hog_feats.append(f)
        return np.array(hog_feats)

    raise ValueError("Unknown feature type")

def train_and_predict(X_train, y_train, X_eval, y_eval, C_value=1.0):
    scaler = StandardScaler()
    # fit scaler exclusively on training data to avoid leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    clf = svm.SVC(kernel='rbf', C=C_value, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_eval_scaled)
    acc = metrics.accuracy_score(y_eval, y_pred)
    return acc, y_pred