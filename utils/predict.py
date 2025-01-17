"""
file : predict.py

author : EPM
cdate : Monday December 2nd 2024
mdate : Monday December 2nd 2024
copyright: 2024 GlobalWalkers.inc. All rights reserved.
"""

from Lib import MnistDataset
import matplotlib.pyplot as plt

mnt = MnistDataset()
LABELS = mnt.labels_map
TEST_SET = mnt.test_set


def predict_loader(model, output_dir, img_count=10):
    model.to("cpu")
    for idx in range(img_count):
        img, label = TEST_SET[idx]
        pred = model(img)
        pred = pred[0].argmax(0)
        plt.title(f"GT: {LABELS[label]}, Predicted: {LABELS[int(pred)]}")
        plt.imshow(img.squeeze())
        plt.savefig(f"{output_dir}/img_{idx}.png")
