"""
file : main.py

author : EPM
cdate : Monday December 2nd 2024
mdate : Monday December 2nd 2024
copyright: 2024 GlobalWalkers.inc. All rights reserved.
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from Lib import MnistDataset, NeuralNetwork
import torch.nn as nn
from utils import train, predict_loader
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import os


def plot_results(hist):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(hist["train_accracy"], label="Training acc")
    plt.plot(hist["val_accuracy"], label="Validation acc")
    plt.legend()
    plt.subplot(122)
    plt.plot(hist["train_loss"], label="Training loss")
    plt.plot(hist["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig("assets/compare_acc.png")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    return loss_fn, optimizer


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config file for training",
    )
    return parser


def main():
    args = make_parser().parse_args()
    cfg = OmegaConf.load(args.config)
    device = get_device()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # dataset and model initialize
    mnist = MnistDataset()
    model = NeuralNetwork().to(device)
    loss_fn, optimizer = get_optimizer(model, cfg["lr"])

    # load dataset
    train_loader, test_loader = mnist.load_dataset(cfg["batch_size"])

    if cfg["training"]:
        # train the model
        hist = train(
            model,
            train_loader,
            test_loader,
            cfg["epochs"],
            cfg["lr"],
            optimizer,
            loss_fn,
            device,
            cfg["model_dir"],
        )
        # compare train and val result with plot and save figure
        plot_results(hist)

    else:
        ckpt = torch.load(
            cfg["ckpt_pth"], map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(ckpt, strict=False)
        if cfg["input_dir"] == "":
            predict_loader(model, cfg["output_dir"])

        # else:
        #     predict_file(model, cfg["input_dir"], cfg["output_dir"])


if __name__ == "__main__":
    main()
