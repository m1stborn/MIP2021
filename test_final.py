import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from configuration import Config
from model.brain_image_dataset import BrainImageDfDataset
from model.metrics import IOU

from utils import *

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation")
    parser.add_argument('--ckpt', type=str, help="Model checkpoint path")

    return parser.parse_args()


if __name__ == '__main__':
    # step 0: Init constants:
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # step 1: Init Network
    ckpt = load_checkpoint(args.ckpt)
    print(vars(ckpt['configs']))
    print(ckpt['args'])
    print(ckpt['trlog'])
    print(ckpt['miou'])

    config = ckpt['configs']
    model = smp.FPN
    if config.model == 'unet++':
        model = smp.UnetPlusPlus
    elif config.model == 'FPN':
        model = smp.FPN

    net = model(
        encoder_name=config.encoder,
        encoder_weights=config.pre_trained_weight,
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    net.to(device)
    net.load_state_dict(ckpt['net'])
    net.eval()

    # step 2: Prepare test data
    test_dataset = BrainImageDfDataset('./archive/test.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False)

    # step 3: Inference
    with torch.no_grad():
        test_metrics = IOU()
        for test_data in test_dataloader:
            images, labels, filenames = test_data[0].to(device), test_data[1], test_data[2]
            outputs = net(images)

            predicted = torch.where(outputs.squeeze(1) > 0.5, 1, 0).cpu().numpy()

            test_metrics.batch_iou(predicted, labels.cpu().numpy())

            # save_overlap_image(filenames, predicted)

        print('Test mIoU: {:.4f}'.format(test_metrics.miou()))

