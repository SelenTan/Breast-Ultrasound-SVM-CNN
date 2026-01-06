import medmnist
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import scipy.ndimage

# global config for dataset path
DATA_FLAG = 'breastmnist'
DATA_ROOT = '../../Datasets'

def get_data_loader(split, augment_mode='none', max_samples_ratio=1.0, random_seed=42):
    # set seed here so results are consistent every time we run this
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

    # simulate low data budget by randomly dropping training samples
    if split == 'train' and max_samples_ratio < 1.0:
        n_samples = int(len(dataset) * max_samples_ratio)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False) #add whole dataset into one batch, I understand this could be used because my dataset is small
    data_batch, labels = next(iter(loader))

    X = data_batch.numpy().squeeze(1)
    y = labels.numpy().ravel()

    # only augment training data to avoid leaking info into validation or test sets
    if split == 'train' and augment_mode != 'none':
        X_aug_list = []
        y_aug_list = []

        """
        Initialize X_temp as a copy of the original X
        This ensures it exists even if we skip rotation
        """
        X_temp = X.copy()

        if augment_mode in ['rotation', 'parallel']:
            X_rot = np.zeros_like(X)
            for i in range(len(X)):
                # rotate image slightly to make model robust to orientation
                angle = np.random.uniform(-15, 15) 
                X_rot[i] = scipy.ndimage.rotate(X[i], angle, reshape=False, mode='nearest')
            X_aug_list.append(X_rot)
            y_aug_list.append(y.copy())

        if augment_mode in ['noise', 'parallel']:
            # add gaussian noise for texture robustness
            noise = np.random.normal(loc=0.0, scale=0.1, size=X.shape)
            X_noise = X_temp + noise
            X_aug_list.append(X_noise)
            y_aug_list.append(y.copy())

        if X_aug_list:
            X_aug_concat = np.concatenate(X_aug_list, axis=0)
            y_aug_concat = np.concatenate(y_aug_list, axis=0)
            # Now we can safely concatenate to the main X and y
            X = np.concatenate((X, X_aug_concat), axis=0)
            y = np.concatenate((y, y_aug_concat), axis=0)

    # flatten images to vectors for the svm
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, y

def extract_features(X_flat, feature_type='raw'):
    if feature_type == 'raw':
        return X_flat

    if feature_type == 'hog':
        # reshape to 2d images to compute gradients then flatten back
        X_imgs = X_flat.reshape(-1, 28, 28)
        hog_feats = []
        for i in range(len(X_imgs)):
            f = hog(
                X_imgs[i],
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                visualize=False
            )
            hog_feats.append(f)
        return np.array(hog_feats)

    raise ValueError("Unknown feature type")

def train_and_predict(X_train, y_train, X_eval, y_eval, C_value=1.0):
    scaler = StandardScaler()
    # fit scaler only on training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    clf = svm.SVC(kernel='rbf', C=C_value, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_eval_scaled)
    acc = metrics.accuracy_score(y_eval, y_pred)
    return acc, y_pred