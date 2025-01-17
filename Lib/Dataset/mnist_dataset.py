"""
file : mnist_dataset.py

author : EPM
cdate : Monday December 2nd 2024
mdate : Monday December 2nd 2024
copyright: 2024 GlobalWalkers.inc. All rights reserved.
"""

import torch
import torch.utils
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class MnistDataset:
    def __init__(self) -> None:
        self.train_set = datasets.MNIST(root="data", train=True, transform=ToTensor())
        self.test_set = datasets.MNIST(root="data", train=False, transform=ToTensor())
        self.train_loader = None
        self.test_loader = None
        self.labels_map = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }

    def load_dataset(self, batch_size: int):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size
        )
        return self.train_loader, self.test_loader

    def display_dataset(self):
        rows, cols = 3, 3
        figure = plt.figure(figsize=(8, 8))
        for n in range(1, (rows * cols) + 1):
            idx = torch.randint(len(self.train_set), size=(1,)).item()
            img, label = self.train_set[idx]
            figure.add_subplot(rows, cols, n)

            plt.title(self.labels_map[label])
            plt.imshow(img.squeeze(), cmap="gray")
            plt.axis("off")
        # plt.show()
        plt.savefig("dataset_img.png")

    def display_loader_images(self):
        rows, cols = 4, 4
        figure = plt.figure(figsize=(8, 8))
        train_images, train_labels = next(iter(self.train_loader))
        for i in range(len(train_images)):
            train_image = train_images[i].squeeze()
            train_label = int(train_labels[i].numpy())
            figure.add_subplot(rows, cols, i + 1)
            plt.title(self.labels_map[train_label])
            plt.imshow(train_image)
            plt.axis("off")
        plt.savefig("dataloader_img.png")


if __name__ == "__main__":
    mnist = MnistDataset()
    batch_size = 16
    train_loader, test_loader = mnist.load_dataset(batch_size)
    mnist.display_dataset()
    mnist.display_loader_images()
