import os

import numpy as np
import cv2
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


class MNISTDataset(Dataset):
    def __init__(self, dataset_path=None, images=None, labels=None):
        if images and labels:
            self.images, self.labels = images, labels
        else:
            self.dataset_path = dataset_path
            self.images, self.labels = self.load_mnist_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        item_dict = {
            "image": Tensor(self.images[item]),
            "label": LongTensor([self.labels[item]])
        }
        return item_dict

    def load_mnist_dataset(self):
        images = list()
        labels = list()

        digit_folder_paths = os.listdir(self.dataset_path)
        folder_count = 1
        for digit in digit_folder_paths:
            digit_images_global_path = self.dataset_path + '/' + digit
            for image_name in os.listdir(digit_images_global_path):
                image_path = digit_images_global_path + '/' + image_name

                image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = np.array(image_gray).flatten()
                label = int(image_path.split(os.path.sep)[-2])

                images.append(image / 255)
                labels.append(label)

            print("[INFO] processed {}/{}".format(folder_count, len(digit_folder_paths)))
            folder_count += 1

        return images, labels


def train_test_dataset_split(dataset, test_size=0.1):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size)
    train_dataset = Subset(dataset, train_idx)
    train_dataset = MNISTDataset(images=[train_dataset[item]['image'].tolist() for item in range(len(train_dataset))],
                                 labels=[train_dataset[item]['label'].item() for item in range(len(train_dataset))])
    test_dataset = Subset(dataset, test_idx)
    test_dataset = MNISTDataset(images=[test_dataset[item]['image'].tolist() for item in range(len(test_dataset))],
                                labels=[test_dataset[item]['label'].item() for item in range(len(test_dataset))])
    return train_dataset, test_dataset
