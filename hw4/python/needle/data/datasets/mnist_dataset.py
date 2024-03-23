from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(label_filename, "rb") as label:
            magic, n = struct.unpack('>2I', label.read(8))
            y = np.frombuffer(label.read(), dtype=np.uint8)

        with gzip.open(image_filename, "rb") as image:
            magic, num, rows, cols = struct.unpack('>4I', image.read(16))
            X = np.frombuffer(image.read(), dtype=np.uint8).reshape(len(y), 784)
    
        X = (X.astype(np.float32) / 255).reshape((-1, 28, 28, 1))
        self.images = X
        self.labels = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.apply_transforms(self.images[index])
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION