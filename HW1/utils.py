import os
import sys
import cv2
import torch
import numpy as np
import skimage.io


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
    with open("./ckpt/log.txt", 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Batch size: {}
Lr:         {} 
Result:
    Epoch:      {}
    Valid IoU:  {}
=======================================================""".format(*args), file=f)


def save_overlap_image(mask_filenames, predicted):
    for i, mask_fn in enumerate(mask_filenames):
        ground_truth = cv2.imread(mask_fn, 0).astype("uint8")
        original_img = cv2.imread(mask_fn.replace("_mask", ""))

        predicted = (predicted * 255.).astype("uint8")

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)

        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_p, _ = cv2.findContours(predicted[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        overlap_img = cv2.drawContours(original_img, contours_gt, 0, (0, 255, 0), 1)
        overlap_img = cv2.drawContours(overlap_img, contours_p, 0, (0, 0, 255), 1)

        cv2.imwrite('./test/'+mask_fn.split('\\')[-1].replace('_mask.tif', '.jpg'),
                    overlap_img)

