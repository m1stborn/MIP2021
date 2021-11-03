import torch
from torch.utils.data import DataLoader, random_split

from model.brain_image_dataset import BrainImageDataset
from model.vgg16_fcn8 import Vgg16FCN8
from model.metrics import IOU
from parse_config import create_parser
from utils import load_checkpoint, save_overlap_image

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    # prepare dataset
    dset = BrainImageDataset('archive/kaggle_3m')
    _, _, test_dataset = random_split(dset, [3005, 393, 531])

    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    ckpt = load_checkpoint(configs.ckpt)

    net = Vgg16FCN8()
    net.load_state_dict(ckpt['net'])
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    with torch.no_grad():
        test_metrics = IOU()
        cnt = 0
        for test_data in test_dataloader:
            images, labels, filenames = test_data[0].to(device), test_data[1], test_data[2]
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            test_metrics.batch_iou(predicted, labels.cpu().numpy())

            save_overlap_image(filenames, predicted)

        print('Test mIoU: {:.4f}'
              .format(test_metrics.miou()))
