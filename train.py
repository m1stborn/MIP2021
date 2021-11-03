import os
import time
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.vgg16_fcn8 import Vgg16FCN8
from model.brain_image_dataset import BrainImageDataset
from model.metrics import IOU
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    pre_val_miou = 0.0

    # step 1: prepare dataset
    dset = BrainImageDataset('archive/kaggle_3m')
    train_dataset, val_dataset, test_dataset = random_split(dset, [3005, 393, 531])

    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    total_steps = len(train_dataloader)

    # step 2: init network
    net = Vgg16FCN8()

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=configs.lr)

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
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

    # step 5: move Net to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    # step 6: main loop
    loss_history = []
    val_history = []

    for epoch in range(start_epoch, start_epoch + configs.epochs):
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
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f}'.format(running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)
            if configs.test_run:
                break

        loss_history.append(running_loss/len(train_dataloader))

        # print Valid mIoU per epoch
        net.eval()
        with torch.no_grad():
            val_metrics = IOU()
            for val_data in val_dataloader:
                images, labels = val_data[0].to(device), val_data[1]
                outputs = net(images)
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                val_metrics.batch_iou(predicted, labels.cpu().numpy())

            print('\nValid mIoU: {:.4f}'
                  .format(val_metrics.miou()))
            val_history.append(val_metrics.miou())

            miou = val_metrics.miou()

            if pre_val_miou < miou:
                checkpoint = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optim': optimizer.state_dict(),
                    'uid': uid,
                    'miou': miou,
                    'loss_his': loss_history,
                    'val_iou_his': val_history
                }
                save_checkpoint(checkpoint,
                                os.path.join(configs.ckpt_path, "Vgg16FCN8-{}.pt".format(uid[:8])))
                # pre_val_miou = val_metrics.mean_iou
                pre_val_miou = miou
                best_epoch = epoch + 1

            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optim': optimizer.state_dict(),
                    'uid': uid,
                    'miou': miou,
                    'loss_his': loss_history,
                    'val_iou_his': val_history
                }
                save_checkpoint(checkpoint,
                                os.path.join(configs.ckpt_path, "Vgg16FCN8-{}-epoch{}.pt".format(uid[:8],epoch+1)))

    # step 7: logging experiment
    print(loss_history)
    print(val_history)
    experiment_record(uid, time.ctime(), configs.batch_size, configs.lr, best_epoch, pre_val_miou)
