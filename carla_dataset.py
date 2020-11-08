from pathlib import Path
from itertools import repeat

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import imgaug.augmenters as iaa

from .const import GAP, STEPS
from .util import rotate_origin_only


np.random.seed(0)
torch.manual_seed(0)

CROP_SIZE = 192
MAP_SIZE = 320

def pad_collate(batch):
    _, M, w, a = zip(*batch)
    M_len = [len(m) for m in M]
    M_mask = torch.zeros(len(M_len), max(M_len), dtype=torch.bool)
    for i, m_len in enumerate(M_len):
        M_mask[i,:m_len] = 1
    M_pad = pad_sequence(M, batch_first=True, padding_value=0)

    return M_pad, M_mask, torch.as_tensor(np.stack(w)), torch.as_tensor(np.stack(a))

def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data

class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        datasets = torch.utils.data.ConcatDataset(data)

        self.dataloader = torch.utils.data.DataLoader(datasets, shuffle=True,
                batch_size=batch_size, num_workers=num_workers, drop_last=True,
                collate_fn=pad_collate, pin_memory=True)
        self.data = repeater(self.dataloader)
        self.samples = samples

    def __iter__(self):
        for _ in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples

def get_dataset(dataset_dir, batch_size=128, num_workers=4, **kwargs):
    def make_dataset(train_or_val):
        data = list()
        transform = transforms.Compose([
            get_augmenter() if train_or_val == 'train' else lambda x: x,
            transforms.ToTensor()
            ])

        episodes = list((Path(dataset_dir) / train_or_val).glob('*'))

        for i, _dataset_dir in enumerate(episodes):
            data.append(CarlaDataset(str(_dataset_dir), transform, **kwargs))

            if i % 5 == 0:
                print(f'{i} episodes')

        data = Wrap(data, batch_size, 250 if train_or_val == 'train' else 25, num_workers)

        return data

    train = make_dataset('train')
    val = make_dataset('val')

    return train, val


def crop_birdview(birdview, dx=0, dy=0):
    x = 238 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Sometimes(0.05, iaa.GaussianBlur((0.0, 1.3))),
        iaa.Sometimes(0.05, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))),
        iaa.Sometimes(0.05, iaa.Dropout((0.0, 0.1))),
        iaa.Sometimes(0.10, iaa.Add((-0.05 * 255, 0.05 * 255), True)),
        iaa.Sometimes(0.20, iaa.Add((0.25, 2.5), True)),
        iaa.Sometimes(0.05, iaa.contrast.LinearContrast((0.5, 1.5))),
        iaa.Sometimes(0.05, iaa.MultiplySaturation((0.0, 1.0))),
        ])

    return seq.augment_image


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), **kwargs):
        dataset_dir = Path(dataset_dir)

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.xy = np.empty((len(list(dataset_dir.glob('image*.png'))), 2))

        self.frames = list()

        for image_path in sorted(dataset_dir.glob('image*.png')):
            frame = str(image_path.stem).split('_')[-1]

            assert (dataset_dir / ('image_%s.png' % frame)).exists()
            assert (dataset_dir / ('segmentation_%s.png' % frame)).exists()
            assert (dataset_dir / ('birdview_%s.npy' % frame)).exists()
            assert (dataset_dir / ('measurements_%s.npy' % frame)).exists()
            x, y, _, _ = np.load(str(dataset_dir / ('measurements_%s.npy' % frame)))
            self.xy[int(frame), :] = np.array([x, y])

            self.frames.append(frame)

    def __len__(self):
        return len(self.frames) - (GAP * (STEPS+1))

    def __getitem__(self, idx):
        path = self.dataset_dir
        frame = self.frames[idx]

        rgb = np.array(Image.open(str(path / ('image_%s.png' % frame))))
        rgb = self.transform(rgb)
        measurements = np.load(str(path / ('measurements_%s.npy' % frame)))

        x, y, z, angle = measurements

        birdview = np.load(str(path / ('birdview_%s.npy' % frame)))
        birdview = np.float32(birdview[4:,4:] != birdview[:-4,:-4])[::4,::4]

        # TODO: add velocity_x, velocity_y
        points = np.empty((int(birdview.sum())+1, 3)) # x, y, class
        points[0] = np.array([20., 0., 0]) # NOTE: ego-vehicle
        s = 1
        for c in range(birdview.shape[-1]):
            l = int(birdview[...,c].sum())
            if l > 0:
                x, y = np.where(birdview[...,c])
                x = (2*(160//4)) - x
                y -= (160//4)
                points[s:s+l,:2] = np.stack([x,y], axis=1)
                points[s:s+l,2] = c + 1 # NOTE: class 0 = ego-vehicle
                s += l
        points = torch.as_tensor(points, dtype=torch.float)

        waypoints = self.xy[idx+GAP:idx+(GAP*(STEPS+1)):GAP] - self.xy[idx]
        waypoints = np.stack(rotate_origin_only(*waypoints.T, angle), axis=-1)

        turn = np.arctan2(waypoints[-1, 1], waypoints[-1, 0])
        action = 1 if turn > 0.10 else (2 if turn < -0.10 else 0)

        return rgb, points, waypoints, action


if __name__ == '__main__':
    import sys
    import cv2
    from .util import visualize_birdview

    data = CarlaDataset(sys.argv[1])
    for i in range(len(data)):
        rgb, points, waypoints, action = data[i]

        canvas = visualize_birdview(points, action, waypoints)
        cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        rgb = np.uint8(255*rgb.permute(1,2,0))
        #cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        cv2.waitKey(10)
