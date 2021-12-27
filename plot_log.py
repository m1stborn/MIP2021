import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from utils import *

if __name__ == '__main__':
    plt.style.use('ggplot')
    # filenames = ['./ckpt/log/',
    #              './ckpt/log/',
    #              './ckpt/log/',
    #              './ckpt/log/']

    # encoder_name = "Vgg"
    # filenames = ['./ckpt/log/FPN-vgg11-a7b71089-end.pt',
    #              './ckpt/log/FPN-vgg13-982b9870-end.pt',
    #              './ckpt/log/FPN-vgg16-68d0473b-end.pt']

    # encoder_name = "Resnet"
    # filenames = ['./ckpt/log/FPN-resnet18-0402d47d-end.pt',
    #              './ckpt/log/FPN-resnet34-d226f81e-end.pt',
    #              './ckpt/log/FPN-resnet50-42974758-end.pt']

    encoder_name = "Efficientnet"
    filenames = ['./ckpt/log/FPN-efficientnet-b3-13391b82-end.pt',
                 './ckpt/log/FPN-efficientnet-b4-fc06bc19-end.pt',
                 './ckpt/log/FPN-efficientnet-b5-d4544078-end.pt',
                 './ckpt/log/FPN-efficientnet-b6-100e60a3-end.pt']


    # Mean IoU:
    fig1 = plt.figure(figsize=(10, 6))
    for filename in filenames:
        ckpt = load_checkpoint(filename)
        e = np.arange(len(ckpt['trlog']['val_miou']))
        config = ckpt['configs']
        plt.plot(e, ckpt['trlog']['val_miou'], label=f'{config.encoder}', lw=3)

    plt.title(f"{encoder_name} Valid mIoU", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("mean IoU", fontsize=15)

    plt.savefig(f"./image/final/{encoder_name}_valid_iou.jpg")

    # Loss:
    fig2 = plt.figure(figsize=(10, 6))
    for filename in filenames:
        ckpt = load_checkpoint(filename)
        e = np.arange(len(ckpt['trlog']['train_loss']))
        config = ckpt['configs']
        plt.plot(e, ckpt['trlog']['train_loss'], label=f'{config.encoder}', lw=3)

    plt.title(f"{encoder_name} Training Loss", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("loss", fontsize=15)

    plt.savefig(f"./image/final/{encoder_name}_train_loss.jpg")
