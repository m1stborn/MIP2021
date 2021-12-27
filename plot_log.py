import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *

if __name__ == '__main__':
    plt.style.use('ggplot')

    # filenames = ['./ckpt/FPN-efficientnet-b7-c7573115.pt',
    #              './ckpt/FPN-efficientnet-b3-e272f484.pt',
    #              './ckpt/FPN-efficientnet-b5-7b243907.pt']

    filenames = ['./ckpt/log/FPN-vgg11-506949a7-end.pt']

    # Mean IoU:
    fig1 = plt.figure(figsize=(10, 6))
    for filename in filenames:
        ckpt = load_checkpoint(filename)
        e = np.arange(len(ckpt['trlog']['val_miou']))
        config = ckpt['configs']
        plt.plot(e, ckpt['trlog']['val_miou'], label=f'{config.encoder}', lw=3)

    plt.title("Efficientnet Valid mIoU", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("mean IoU", fontsize=15)

    plt.savefig("./image/final/efficientnet_valid_iou.jpg")

    # Loss:
    fig2 = plt.figure(figsize=(10, 6))
    for filename in filenames:
        ckpt = load_checkpoint(filename)
        e = np.arange(len(ckpt['trlog']['train_loss']))
        config = ckpt['configs']
        plt.plot(e, ckpt['trlog']['train_loss'], label=f'{config.encoder}', lw=3)

    plt.title("Efficientnet Training Loss", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("loss", fontsize=15)

    plt.savefig("./image/final/efficientnet_train_loss.jpg")
