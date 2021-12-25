import os
import csv
import time
import uuid
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from configuration import Config
from model.brain_image_dataset import BrainImageDataset, BrainImageDfDataset
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
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path args
    parser.add_argument('--ckpt_path', default='./ckpt', type=str,
                        help="Model checkpoint path")

    parser.add_argument('--image_dir', default='./archive/kaggle_3m', type=str,
                        help="Images dir for Train/Valid/Test")

    return parser.parse_args()


def activate_relu(**params):
    return nn.ReLU()


if __name__ == '__main__':
    # step 0: Init constants:
    config = Config()
    args = parse_args()
    uid = str(uuid.uuid1())
    best_epoch = 0
    prev_val_miou = 0.0

    # filelist_df = pd.read_csv('./archive/filelist.csv')
    # train_df, test_df = train_test_split(filelist_df, test_size=0.25)
    # test_df, valid_df = train_test_split(test_df, test_size=0.4)

    # train_df.to_csv('./archive/train.csv', index=False)
    # valid_df.to_csv('./archive/valid.csv', index=False)
    # test_df.to_csv('./archive/test.csv', index=False)
    # train: 75% = 2946, valid: 10% = 394, test: 15% = 589

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    trlog = dict()
    trlog['train_loss'] = []
    trlog['val_miou'] = []

    # step 2: Init network
    net.to(device)
    # from torchsummary import summary
    # summary(net, (3, 256, 256))

    # net2 = Vgg16FCN8()
    # net2.to(device)
    # x = torch.randn(1, 3, 256, 256).to(device)
    # out = net2(x)
    # print(out.size())

    mem_params = sum([param.nelement() * param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(f"GPU Mem (MB): {mem / 1000000}")

    # step 1: Prepare dataset
    dset = BrainImageDataset('archive/kaggle_3m')

    # train_dataset, val_dataset, test_dataset = random_split(dset, [3005, 393, 531])
    train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.RandomAutocontrast(p=0.3),
                transforms.ColorJitter(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    train_dataset = BrainImageDfDataset('./archive/train.csv', transform=train_transform)
    val_dataset = BrainImageDfDataset('./archive/valid.csv')
    test_dataset = BrainImageDfDataset('./archive/test.csv')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False)

    total_steps = len(train_dataloader)

    # step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # step 4: Check if resume training
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.ckpt)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        pre_val_miou = ckpt['miou']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 6: Main loop
    for epoch in range(start_epoch, start_epoch + config.epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + config.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f}'.format(running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)
            if args.test_run:
                break

        trlog['train_loss'].append(running_loss / len(train_dataloader))

        # print Valid mIoU per epoch
        net.eval()
        with torch.no_grad():
            val_metrics = IOU()
            for val_data in val_dataloader:
                images, labels = val_data[0].to(device), val_data[1]
                outputs = net(images)
                # Categorical
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                val_metrics.batch_iou(predicted, labels.cpu().numpy())

            print('\nValid mIoU: {:.4f}'
                  .format(val_metrics.miou()))
            trlog['val_miou'].append(val_metrics.miou())

            miou = val_metrics.miou()

            if args.test_run:
                break

            if prev_val_miou < miou:
                checkpoint = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optim': optimizer.state_dict(),
                    'uid': uid,
                    'miou': miou,
                    'configs': config,
                    'args': vars(args),
                    'trlog': trlog
                }
                ckpt_name = "FPN-{}-{}.pt".format(config.encoder, uid[:8])
                save_checkpoint(checkpoint, os.path.join(args.ckpt_path, ckpt_name))

                prev_val_miou = miou
                best_epoch = epoch + 1

    # step 7: Logging experiment
    if not args.test_run:
        logger('./ckpt/log_final.txt',
               uid,
               time.ctime(),
               best_epoch,
               prev_val_miou)
