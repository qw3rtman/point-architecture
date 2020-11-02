from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 4
STEPS = 5

COMMANDS = 6

CROP_SIZE = 192
MAP_SIZE = 320
BIRDVIEW_CHANNELS = 7

def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data

class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        datasets = torch.utils.data.ConcatDataset(data)

        self.dataloader = torch.utils.data.DataLoader(datasets, shuffle=True,
                batch_size=batch_size, num_workers=num_workers, drop_last=True,
                pin_memory=True)
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
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episodes)))

        for _dataset_dir in episodes[:num_episodes]:
            data.append(CarlaDataset(str(_dataset_dir), transform, **kwargs))

        print('%d frames.' % sum(map(len, data)))

        # TODO: batching via masking!!
        data = torch.utils.data.ConcatDataset(data)
        data = Wrap(data, batch_size, 1000 if train_or_val == 'train' else 100, num_workers)

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


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), **kwargs):
        dataset_dir = Path(dataset_dir)

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.frames = list()

        for image_path in sorted(dataset_dir.glob('image*.png')):
            frame = str(image_path.stem).split('_')[-1]

            assert (dataset_dir / ('image_%s.png' % frame)).exists()
            assert (dataset_dir / ('segmentation_%s.png' % frame)).exists()
            assert (dataset_dir / ('birdview_%s.npy' % frame)).exists()
            assert (dataset_dir / ('measurements_%s.npy' % frame)).exists()

            self.frames.append(frame)

    def __len__(self):
        return len(self.frames) - GAP * STEPS

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
        points = np.empty((int(birdview.sum()), 3)) # x, y, class
        s = 0
        for c in range(birdview.shape[-1]):
            l = int(birdview[...,c].sum())
            if l > 0:
                points[s:s+l,:2] = np.stack(np.where(birdview[...,c]), axis=1)
                points[s:s+l,2] = c
                s += l
        points = torch.as_tensor(points, dtype=torch.float)

        return rgb, points, '%s %s' % (path.stem, frame)


if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw
    from ..utils.common import visualize_birdview
    from ..utils.converter import ConverterTorch

    data = CarlaDataset(sys.argv[1])
    convert = ConverterTorch()

    for i in range(len(data)):
        rgb, birdview, meta = data[i]
        canvas = np.uint8(birdview.detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas = visualize_birdview(canvas)
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)

        origin = np.array([birdview.shape[-1] // 2, birdview.shape[-1] - 5])
        offsets = np.array([[0, 0], [0, -10], [-5, -20]])
        points = origin + offsets
        points_cam = convert(torch.FloatTensor(points))
        points_reproject = convert.cam_to_map(points_cam)

        print(points)

        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))

        for x, y in points_reproject:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))

        canvas_rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas_rgb = Image.fromarray(canvas_rgb)
        draw_rgb = ImageDraw.Draw(canvas_rgb)

        for x, y in points_cam:
            draw_rgb.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))


        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.imshow('rgb', cv2.cvtColor(np.array(canvas_rgb), cv2.COLOR_BGR2RGB))
        cv2.waitKey(200)
