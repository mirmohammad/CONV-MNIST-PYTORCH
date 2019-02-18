import gzip
import pickle

import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class MNIST(Dataset):

    def __init__(self, root, split, transform=None):
        if split == 'train':
            f = gzip.open(root, 'rb')
            (self.images, self.labels), _, _ = pickle.load(f, encoding='latin1')
            self.num_images = self.images.shape[0]
        elif split == 'valid':
            f = gzip.open(root, 'rb')
            _, (self.images, self.labels), _ = pickle.load(f, encoding='latin1')
            self.num_images = self.images.shape[0]
        elif split == 'test':
            f = gzip.open(root, 'rb')
            _, _, (self.images, self.labels) = pickle.load(f, encoding='latin1')
            self.num_images = self.images.shape[0]
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.images[idx, :]
        label = self.labels[idx]

        image = np.reshape(image, (28, 28))

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
