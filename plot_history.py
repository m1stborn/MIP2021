import numpy as np
import matplotlib.pyplot as plt

from parse_config import create_parser
from utils import load_checkpoint, save_overlap_image


if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    ckpt = load_checkpoint(configs.ckpt)

    # plotting
    plt.style.use('ggplot')
    e = np.arange(ckpt['epoch']+1)

    print(len(ckpt['loss_his']), len(ckpt['val_iou_his']))

    # plot training loss history
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, ckpt['loss_his'], label='train loss', lw=3, c="tab:blue")

    plt.title("Vgg16FCN8", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.savefig("training_loss_history.jpg")

    # plot valid IoU history
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(e, ckpt['val_iou_his'], label='valid IoU', lw=3, c="tab:orange")

    plt.title("Vgg16FCN8", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("IoU", fontsize=15)

    plt.savefig("valid_IoU_history.jpg")
