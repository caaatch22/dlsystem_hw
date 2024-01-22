from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct

# class MNISTDataset(Dataset):
#     def __init__(
#         self,
#         image_filename: str,
#         label_filename: str,
#         transforms: Optional[List] = None,
#     ):
#         ### BEGIN YOUR SOLUTION
        # self.transforms = transforms
        # with gzip.open(image_filename, "rb") as img_file:
        #     magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
        #     assert(magic_num == 2051)
        #     tot_pixels = row * col
        #     imgs = [np.array(struct.unpack(f"{tot_pixels}B",
        #                                    img_file.read(tot_pixels)),
        #                                    dtype=np.float32)
        #             for _ in range(img_num)]
        #     X = np.vstack(imgs)
        #     X -= np.min(X)
        #     X /= np.max(X)
        #     self.X = X
        # with gzip.open(label_filename, "rb") as label_file:
        #     magic_num, label_num = struct.unpack(">2i", label_file.read(8))
        #     assert(magic_num == 2049)
        #     self.y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
#         ### END YOUR SOLUTION

#     def __getitem__(self, index) -> object:
#         ### BEGIN YOUR SOLUTION
#         imgs = self.X[index]
#         labels = self.y[index]
#         if len(imgs.shape) > 1:
#             imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
#         else:
#             imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
#         return (imgs, labels)
#         ### END YOUR SOLUTION

#     def __len__(self) -> int:
#         ### BEGIN YOUR SOLUTION
#         return self.X.shape[0]
        ### END YOUR SOLUTION

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