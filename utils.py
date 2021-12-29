import os
import sys
import csv
import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_checkpoint(state, save_path: str):
    torch.save(state, save_path)


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt


def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    # percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(prefix + '[%s]-Step [%s/%s]-%s\r' % (bar, count, total, suffix))
    sys.stdout.flush()

    # if count == total:
    #     print("\n")


def experiment_record(*args):
    with open("ckpt/log.txt", 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Batch size: {}
Lr:         {} 
Result:
    Epoch:      {}
    Valid IoU:  {}
=======================================================""".format(*args), file=f)


def logger(fn, *args):
    with open(fn, 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Checkpoint: {}
Result:
    Epoch:      {}
    Valid IoU:  {}
=======================================================""".format(*args), file=f)


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def dice_coef_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)


def bce_dice_loss(inputs, target, alpha=0.01):
    dice_loss = dice_coef_loss(inputs, target)
    bce_criterion = nn.BCELoss()
    bce_loss = bce_criterion(inputs, target)

    return bce_loss + alpha * dice_loss


def iou_loss(inputs, target):
    smooth = 1.0
    inputs = inputs.view(-1)
    target = target.view(-1)

    intersection = (inputs * target).sum()
    total = (inputs + target).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)

    return 1 - iou


def bce_iou_loss(inputs, target, alpha=0.01):
    iou = iou_loss(inputs, target)
    bce_criterion = nn.BCELoss()
    bce_loss = bce_criterion(inputs, target)

    return bce_loss + alpha * iou


def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))

    iou = (intersection + e) / (union + e)
    return iou


def create_data_csv(filepath):
    rows = []
    for (rootdir, subdir, filenames) in os.walk(filepath):
        for f in filenames:
            if not f.endswith('mask.tif'):
                img_filename = os.path.join(rootdir, f)
                mask_filename = os.path.join(rootdir, f.replace('.tif', '_mask.tif'))
                rows.append((img_filename, mask_filename))

    with open('./archive/filelist.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'image', 'label'])
        for i, row in enumerate(rows):
            writer.writerow([i] + list(row))


def save_overlap_image(mask_filenames, pred):
    """
    Saving original image as .jpg and save prediction with ground truth
    :param mask_filenames:
    :param pred:
    :return:
    """
    masks_rgb = np.empty((len(pred), 256, 256, 3))
    for i, p in enumerate(pred):
        masks_rgb[i, p == 1] = [255, 255, 255]  # (White: 111) tumor
        masks_rgb[i, p == 0] = [0, 0, 0]  # (Black: 000) Not tumor
    masks_rgb = masks_rgb.astype(np.uint8)

    for i, mask_fn in enumerate(mask_filenames):
        ground_truth = cv2.imread(mask_fn, 0).astype("uint8")
        original_img = cv2.imread(mask_fn.replace("_mask", ""))

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours_p, _ = cv2.findContours(pred[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imwrite('./test/' + mask_fn.split('\\')[-1].replace('_mask.tif', '.jpg'),
                    original_img)

        overlap_mask_gt = cv2.drawContours(masks_rgb[i], contours_gt, 0, (0, 255, 0), 1)

        cv2.imwrite('./test/' + mask_fn.split('\\')[-1].replace('.tif', '_gt.jpg'),
                    overlap_mask_gt)


def save_overlap_image_combine(mask_filenames, pred):
    width = 2
    columns = 1
    fig, axs = plt.subplots(columns, width, figsize=(16 * width, 16 * columns), constrained_layout=True)
    red_patch = mpatches.Patch(color='red', label='The red data')
    fig.legend(loc='upper right', handles=[
        mpatches.Patch(color='red', label='Ground truth'),
        mpatches.Patch(color='green', label='Predicted abnormality')])
    for i, mask_fn in enumerate(mask_filenames):
        original_img = cv2.imread(mask_fn.replace("_mask", ""))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        axs[0].imshow(original_img)
        axs[0].title.set_text('Brain MRI')

        original_img[pred[i].cpu().bool()] = (0, 255, 150)
        ground_truth = cv2.imread(mask_fn, 0).astype("uint8")

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlap_mask_gt = cv2.drawContours(original_img, contours_gt, 0, (255, 0, 0), 1)

        axs[1].imshow(overlap_mask_gt)
        axs[1].title.set_text('Predicted and Ground Truth')

        filename = mask_fn.split('\\')[-1].replace('.tif', '.jpg')
        plt.savefig(f'./image/final/all/{filename}')
