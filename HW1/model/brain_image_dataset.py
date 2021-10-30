import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class BrainImageDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        if transform is not None:
            self.transform = transform

        for (rootdir, subdir, filenames) in os.walk(self.root):
            for f in filenames:
                if not f.endswith('mask.tif'):
                    img_filename = os.path.join(rootdir, f)
                    mask_filename = os.path.join(rootdir, f.replace('.tif', '_mask.tif'))
                    self.filenames.append((img_filename, mask_filename))

        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_filename, mask_filename = self.filenames[idx]
        img = cv2.imread(img_filename)
        img = self.transform(img)

        mask = cv2.imread(mask_filename)  # 256, 256 ,3
        mask = read_mask(mask)

        return img, torch.tensor(mask).long(), mask_filename

    # def sample(self, idx):
    #     img_filename, mask_filename = self.filenames[idx]
    #     image = cv2.imread(img_filename)
    #     image = np.array(image) / 255.
    #     mask = cv2.imread(mask_filename, 0)
    #     mask = np.array(mask) / 255.
    #
    #     image = image.transpose((2, 0, 1))
    #     image = torch.from_numpy(image).type(torch.float32)
    #     image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    #     mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
    #     mask = torch.from_numpy(mask).type(torch.float32)
    #
    #     return image, mask

    def __len__(self):
        return self.len


def read_mask(mask):
    out = np.empty((mask.shape[0], mask.shape[1]))

    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

    out[mask == 0] = 0  # (Black: 000) Not tumor
    out[mask == 7] = 1  # (White: 111) tumor
    return out


if __name__ == '__main__':
    train_dataset = BrainImageDataset('../archive/kaggle_3m')
    print(len(train_dataset))
    # im, ma = train_dataset[5]
    # print(im.size(), ma.size())
    # print(im[1, :, :])
    # print(im.permute(1, 2, 0).size())
    # import matplotlib.pyplot as plt
    # plt.imshow(im.permute(1, 2, 0))
    # plt.show()

    # im2, ma2 = train_dataset.sample(5)
    # print(im2.size(), ma.size())

    # TODO: check if mask correct
    # print(np.unique(ma2.cpu().numpy() == 1.))
    # print(np.unique(ma.cpu().numpy() == 1.))
    # print(np.array_equal(ma.cpu().numpy() == 1., ma2.cpu().numpy() == 1.))

    # show = np.zeros((256, 256, 3))
    # show2 = np.zeros((256, 256, 3))
    # show[:, :, 2] = ma
    # show2[:, :, 2] = ma2
    # show = show * 255
    # show2 = show2 * 255
    # plt.imshow(show)
    # plt.show()
    # plt.imshow(show2)
    # plt.show()

    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    #
    # from torch.utils.data import DataLoader, random_split
    # dset = BrainImageDataset('./archive/kaggle_3m')
    # train_dataset, val_dataset, test_dataset = random_split(dset, [3005, 393, 531])
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=32,
    #                               shuffle=True)
    #
    # val_dataloader = DataLoader(val_dataset, batch_size=32,
    #                             shuffle=True)
    #
    # test_dataloader = DataLoader(test_dataset, batch_size=32,
    #                              shuffle=False)
    # batch = next(iter(test_dataloader))
    # for fn in batch[2]:
    #     print(fn.split('\\')[-1])
