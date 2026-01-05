import medmnist
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import scipy.ndimage  # used for rotation on numpy images

# dataset config
DATA_FLAG = 'breastmnist'
DATA_ROOT = '../../Datasets'


def get_data_loader(split, augment_mode='none', max_samples_ratio=1.0):
    # loads a medmnist split and returns numpy arrays for classical models
    # augmentation is done in numpy so we can keep the rest of the pipeline simple
    info = medmnist.INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])

    # basic tensor pipeline then convert back to numpy for svm
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

    if split == 'train' and max_samples_ratio < 1.0:
        # subsample the training split to simulate limited data budget
        n_samples = int(len(dataset) * max_samples_ratio)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = data.Subset(dataset, indices)

    # pull the whole split into memory in one go
    loader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data_batch, labels = next(iter(loader))

    # turn into numpy and drop the channel dimension
    X = data_batch.numpy().squeeze(1)
    y = labels.numpy().ravel()

    if split == 'train' and augment_mode != 'none':
        # build extra samples then append them to the original train set
        X_aug_list = []
        y_aug_list = []

        if augment_mode in ['rotation', 'combined']:
            # rotate each image by a random angle, keep output size 28x28
            X_rot = np.zeros_like(X)
            for i in range(len(X)):
                angle = np.random.uniform(-15, 15) #Random Rotation -15 to +15 degrees
                X_rot[i] = scipy.ndimage.rotate(X[i], angle, reshape=False, mode='nearest')

            X_aug_list.append(X_rot)
            y_aug_list.append(y)

        if augment_mode in ['noise', 'combined']:
            # add gaussian noise then clip back to a reasonable value range
            noise = np.random.normal(loc=0.0, scale=0.1, size=X.shape)
            X_noise = X + noise
            X_noise = np.clip(X_noise, X.min(), X.max())

            X_aug_list.append(X_noise)
            y_aug_list.append(y)

        if X_aug_list:
            # concatenate augmented samples, then concatenate with the original set
            X_aug_concat = np.concatenate(X_aug_list, axis=0)
            y_aug_concat = np.concatenate(y_aug_list, axis=0)

            X = np.concatenate((X, X_aug_concat), axis=0)
            y = np.concatenate((y, y_aug_concat), axis=0)

    # flatten to N,784 for classical models
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, y


def extract_features(X_flat, feature_type='raw'):
    # feature switch used by the experiment runner
    # raw keeps pixels, hog builds a hand crafted descriptor per image
    if feature_type == 'raw':
        return X_flat

    if feature_type == 'hog':
        # reshape back to images then extract hog per sample
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


def train_and_predict(X_train, y_train, X_test, y_test, C_value=1.0):
    # standardize features before svm, especially important for rbf kernel
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # rbf svm, C controls the regularization strength
    clf = svm.SVC(kernel='rbf', C=C_value, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc, y_pred
