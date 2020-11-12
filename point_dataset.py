from pathlib import Path
from itertools import repeat
from operator import attrgetter

import numpy as np
import torch
from joblib import Memory
memory = Memory('/scratch/cluster/nimit/cache', verbose=0)
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageDraw, ImageFont

from .const import GAP, STEPS
from .util import rotate_origin_only, BACKGROUND, COLORS, ACTIONS

import sys
sys.path.append('/u/nimit/Documents/robomaster/point_policy')
from point.recording import parse
parse_recording = memory.cache(parse)
from point.visualization import follow_view_matrix


CROP_SIZE = 192
MAP_SIZE = 320

def pad_collate(batch):
    M, a, v, w = zip(*batch)
    M_len = [len(m) for m in M]
    M_mask = torch.zeros(len(M_len), max(M_len), dtype=torch.bool)
    for i, m_len in enumerate(M_len):
        M_mask[i,:m_len] = 1
    M_pad = pad_sequence(M, batch_first=True, padding_value=0)

    return M_pad, M_mask, torch.as_tensor(np.stack(a)), torch.as_tensor(np.stack(v), dtype=torch.float).unsqueeze(dim=1), torch.as_tensor(np.stack(w))

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

        episodes = list((Path(dataset_dir) / train_or_val).glob('*'))
        for i, _dataset_dir in enumerate(episodes):
            data.append(PointDataset(str(_dataset_dir), **kwargs))

            if i % 5 == 0:
                print(f'{i} episodes')

        return Wrap(data, batch_size, 250 if train_or_val == 'train' else 25, num_workers)

    train = make_dataset('train')
    val = make_dataset('val')

    return train, val


class PointDataset(Dataset):

    def __init__(self, dataset_dir):
        self.world_map, self.frames = parse_recording(dataset_dir)
        self.frames = self.frames[::10] # 20 FPS -> 2 Hz
        self.ego_uid, self.ego_id = attrgetter('uid', 'id')(self.frames[0].cars[0])

        self.xy = np.stack([f.actor_by_id(self.ego_id).location[:2] for f in self.frames])
        self.get_view_matrix = follow_view_matrix(self.ego_id, 100, 100, rotate=True)

    def __len__(self):
        return len(self.frames) - (GAP * (STEPS+1))

    def __getitem__(self, idx):
        points = []
        # TODO: prune actors that are too far away
        for actor in self.frames[idx].actors:
            c = self._get_class(actor)
            if c is not None:
                points.append(np.concatenate([actor.location[:2], np.array([c])]))

        points = np.stack(points)
        view_matrix = self.get_view_matrix(self.world_map, self.frames[idx], []).reshape(3,2)
        xy = np.concatenate([points[:,:2], np.ones((points.shape[0],1))], axis=-1)
        points[:,:2] = xy @ view_matrix
        points = torch.as_tensor(points, dtype=torch.float)

        waypoints = np.concatenate([
            self.xy[idx+GAP:idx+(GAP*(STEPS+1)):GAP],
            np.ones((STEPS, 1))], axis=-1)
        waypoints = waypoints @ view_matrix.reshape(3,2)
        waypoints = torch.as_tensor(waypoints, dtype=torch.float)

        # TODO: action (see LbC)
        turn = np.arctan2(*waypoints[-1])
        action = 1 # FORWARD
        if np.linalg.norm(waypoints[-1]) < 0.05:
            action = 0 # STOP
        elif turn < -0.20:
            action = 2 # LEFT
        elif turn > 0.20:
            action = 3 # RIGHT

        # TODO: velocity (see LbC)
        velocity = np.linalg.norm((self.xy[idx]-self.xy[idx-1]) if idx > 0 else self.xy[idx+1]-self.xy[idx])

        return points, action, velocity, waypoints

    def _get_class(self, actor):
        if actor.type == 1: # vehicle
            return 0 if actor.id == self.ego_id else 6
        elif actor.type == 2: # pedestrian
            return 7
        elif actor.type == 3: # traffic light
            return actor.state + 3

        return None


font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 16)
def visualize_birdview(points, action, waypoints, _waypoints=None, h=100, r=0.5, w_r=0.5):
    def _scale_points(points):
        return (points + torch.Tensor([1.0,1.0,0.0])) * torch.Tensor([h//2,h//2,1])
    def _scale_waypoints(waypoints):
        return (waypoints + 1.0) * (h//2)

    canvas = np.zeros((h, h, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)

    for x, y, c in _scale_points(points):
        draw.ellipse((x - r, h-y - r, x + r, h-y + r), fill=COLORS[int(c.item())])

    for x, y in _scale_waypoints(waypoints):
        draw.ellipse((x - r, h-y - r, x + r, h-y + r), fill=(0, 175, 0))

    if _waypoints is not None:
        for x, y in _scale_waypoints(_waypoints):
            draw.ellipse((x - r, h-y - r, x + r, h-y + r), fill=(175, 0, 0))

    draw.text((0, 0), ACTIONS[action], fill='black', font=font)
    return canvas


if __name__ == '__main__':
    import sys
    import cv2
    from .const import BACKGROUND, COLORS, ACTIONS
    from PIL import Image, ImageDraw, ImageFont
    #import matplotlib.pyplot as plt

    data = PointDataset(sys.argv[1])
    for i in range(len(data)):
        points, action, velocity, waypoints = data[i]

        canvas = visualize_birdview(points, action, waypoints, h=200, r=1.0)
        cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        cv2.waitKey(10)

        """
        plt.scatter(*points[:,:2].T, s=5, c=points[:,2])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        """
