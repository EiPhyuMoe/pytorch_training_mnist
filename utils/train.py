"""
file : train.py

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


def train_epoch(
    model,
    dataloader,
    lr=0.01,
    optimizer=None,
    loss_fn=nn.NLLLoss(),
    device="",
):
    model.train()
    size = len(dataloader.dataset)
    batch_size = len(dataloader)
    correct, t_loss = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()

        loss = loss_fn(y_pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
        t_loss += loss

    correct /= size
    t_loss /= size
    return model, t_loss.to("cpu").detach(), correct


def validate(model, dataloader, loss_fn=nn.NLLLoss(), device=""):
    model.eval()
    test_loss, correct = 0, 0
    batch_size = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y)
            correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= batch_size
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss.to("cpu").detach(), correct


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    optimizer,
    loss_fn,
    device,
    output_dir,
):
    hist = {
        "train_loss": [],
        "train_accracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(epochs):
        model, train_loss, train_acc = train_epoch(
            model,
            train_loader,
            lr=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_loss, val_acc = validate(model, test_loader, loss_fn=loss_fn, device=device)
        # print(
        #     f"Epoch {epoch:2}, Train acc={train_acc:.3f}, Val acc={val_accuracy:.3f}, Train loss={train_loss:.3f}, Val loss={val_loss:.3f}"
        # )
        hist["train_loss"].append(train_loss)
        hist["train_accracy"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_accuracy"].append(val_acc)
        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")
    return hist
