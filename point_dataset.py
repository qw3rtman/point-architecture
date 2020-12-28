from pathlib import Path
from itertools import repeat, chain
from functools import partial
from operator import itemgetter, attrgetter
import random
import json

import numpy as np
import torch
from joblib import Memory
memory = Memory('/scratch/cluster/nimit/cache', verbose=0)
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageDraw, ImageFont

from .const import MAP_SIZE, GAP, STEPS, BACKGROUND, COLORS, ACTIONS, LANDMARKS
from .util import to_meters_per_second, rotate_origin_only

import sys
sys.path.append('/u/nimit/Documents/robomaster/point_policy')
from point.recording import parse

def get_mask(L):
    mask = torch.zeros(len(L), max(L), dtype=torch.bool)
    for i, l in enumerate(L):
        mask[i,:l] = 1
    return mask

def pad_collate(batch, diameter):
    M, A, W, S, T, B = zip(*batch)

    # prune map to intermediate size, so that step() doesn't get cut off
    # TODO: 2*diameter for step
    M = [prune(m, diameter) for m in M]

    M_mask = get_mask([len(m) for m in M])
    M_pad = pad_sequence(M, batch_first=True, padding_value=0)

    return M_pad, M_mask, torch.as_tensor(np.stack(A)), torch.as_tensor(np.stack(W)), torch.as_tensor(np.stack(S)), torch.as_tensor(np.stack(T)), torch.as_tensor(np.stack(B)),

def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data

class Wrap(object):
    def __init__(self, data, batch_size, map_size, samples, num_workers):
        datasets = torch.utils.data.ConcatDataset(data)

        self.dataloader = torch.utils.data.DataLoader(datasets, shuffle=True,
                batch_size=batch_size, num_workers=num_workers, drop_last=True,
                pin_memory=True, collate_fn=partial(pad_collate, diameter=map_size))
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
        random.shuffle(episodes)
        for i, _dataset_dir in enumerate(episodes):
            if _dataset_dir.is_dir():
                data.append(get_episode(str(_dataset_dir)))

                if i % 5 == 0:
                    print(f'{i} episodes')

        return Wrap(data, batch_size, kwargs.get('map_size', MAP_SIZE),
            250 if train_or_val == 'training' else 25, num_workers)

    train = make_dataset('training')
    val = make_dataset('testing')

    return train, val


@memory.cache()
def get_episode(episode_dir):
    # loads huge map (MAP_SIZE=100)
    return PointDataset(str(episode_dir))


class PointDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)

        self.world_map, self.frames = parse(self.dataset_dir/'recording.log')
        self.frames = self.frames[::10] # 20 FPS -> 2 Hz
        self.ego_uid, self.ego_id = attrgetter('uid', 'id')(self.frames[0].cars[0])

        self.map_waypoints = self.world_map.generate_waypoints(2)
        self.map_landmarks = chain(*[self.world_map.get_all_landmarks_of_type(t) for t in LANDMARKS.keys()])

        self.xy = np.empty((len(self.frames), 2))
        self.rot = np.empty(len(self.frames))
        for idx, frame in enumerate(self.frames):
            ego = frame.actor_by_id(self.ego_id)
            self.xy[idx] = ego.location[:2]
            self.rot[idx] = ego.rotation[2]

        self.rot = np.deg2rad(self.rot)

        # cache frames
        self.points, self.actions, self.waypoints, self.steer, self.throttle, self.brake = [], torch.empty(len(self)), torch.empty(len(self), STEPS, 2), torch.empty(len(self)), torch.empty(len(self)), torch.empty(len(self))
        for idx in range(len(self)):
            _points, _action, _waypoints, _steer, _throttle, _brake = self._get_item(idx)
            self.points.append(_points)
            self.actions[idx], self.waypoints[idx] = _action, _waypoints
            self.steer[idx], self.throttle[idx], self.brake[idx] = _steer, _throttle, _brake

        del self.world_map, self.frames, self.map_waypoints, self.map_landmarks

    def __len__(self):
        if hasattr(self, 'frames'):
            return len(self.frames) - (GAP * (STEPS+1))
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.actions[idx], self.waypoints[idx], self.steer[idx], self.throttle[idx], self.brake[idx]

    def _get_item(self, idx):
        points = []

        # dynamic agents (vehicles, pedestrians)
        for actor in self.frames[idx].actors:
            if np.max(np.abs(actor.location[:2] - self.xy[idx])) <= MAP_SIZE:
                point = self.get_actor_point(actor)
                if point is not None:
                    points.append(point)

        # driving waypoints (road)
        for waypoint in self.map_waypoints:
            location = attrgetter('x','y')(waypoint.transform.location)
            if np.max(np.abs(location - self.xy[idx])) <= MAP_SIZE:
                points.append(np.array([*location, waypoint.transform.rotation.yaw, -1, 0.0, 0]))

        # landmarks (signs)
        for landmark in self.map_landmarks:
            location = attrgetter('x','y')(landmark.transform.location)
            if np.max(np.abs(location - self.xy[idx])) <= MAP_SIZE:
                v = to_meters_per_second(landmark.value, landmark.unit) if landmark.type == '274' else 0.0
                point = np.array([*location, landmark.transform.rotation.yaw, -1, v, LANDMARKS[landmark.type]])

        points = np.stack(points)
        up, right = attrgetter('forward', 'right')(self.frames[idx].actor_by_id(self.ego_id))
        view_matrix = np.array([
            *right[:2], *up[:2],
            -right[:2].dot(self.xy[idx]),
            -up[:2].dot(self.xy[idx])
        ]).reshape(3,2)

        points[:,:2] = np.concatenate([points[:,:2], np.ones((points.shape[0],1))], axis=-1) @ view_matrix
        shift = points[points[:,-1]==1][:,:2].copy()
        points[:,:2] -= shift

        # relative orientation to ego-vehicle
        orientation = np.deg2rad(points[:,2]) - self.rot[idx]
        points[:,2] = np.cos(orientation)
        points[:,3] = np.sin(orientation)

        points = torch.as_tensor(points, dtype=torch.float)

        waypoints = np.concatenate([
            self.xy[idx+GAP:idx+(GAP*(STEPS+1)):GAP],
            np.ones((STEPS, 1))], axis=-1)
        waypoints = torch.as_tensor(waypoints @ view_matrix, dtype=torch.float)
        waypoints -= shift

        with open(self.dataset_dir / f'measurements/{idx:04}.json', 'r') as f:
            measurements = json.load(f)
            action = int(measurements['command']) - 1
            steer, throttle, brake = itemgetter('steer', 'throttle', 'brake')(measurements)

        return points, action, waypoints, steer, throttle, brake

    def get_actor_point(self, actor):
        out = np.empty(6)
        out[:2] = actor.location[:2]
        out[2] = actor.rotation[2]
        out[3] = -1

        if actor.type == 1: # vehicle
            c = 1 if actor.id == self.ego_id else 2
            s = np.linalg.norm(actor.linear_velocity)
        elif actor.type == 2: # pedestrian
            c = 3
            s = np.linalg.norm(actor.linear_velocity)
        elif actor.type == 3: # traffic light
            c = actor.state + 4
            s = 0
            out[2] += 90 # orient toward relevant traffic
        else:
            return None

        out[4] = s
        out[-1] = c

        return out


def prune(points, diameter):
    # points: N x 6
    _points = points[points[...,:2].abs().max(dim=1)[0] <= diameter/2]
    return _points


def batched_prune(points, mask, diameter):
    # NOTE: this will blow up the computation graph...
    # TODO: just sort `points` by max(x, y) dist!! very fast since
    #       pytorch just treats this as a shuffle of indices
    M = [prune(p, diameter) for p in points]

    M_mask = get_mask([len(m) for m in M])
    M_pad = pad_sequence(M, batch_first=True, padding_value=0)

    return M_pad, M_mask


def step(points, waypoints, j, waypoints_gt=None):
    """
    step to waypoints[:,j] for j = 0,...,STEPS-1
    """

    if j == 0:
        return points, waypoints

    dx, dy = (waypoints[:,j] - waypoints[:,max(0,j-1)]).T
    movement = (torch.abs(dx) >= 1e-2) & (torch.abs(dy) >= 1e-2)
    heading = (torch.atan2(dy, dx) - (np.pi/2))
    heading[~movement] = 0.

    xy = points[...,:2].clone()
    # assumes that each batch has exactly 1 ego-vehicle (class=1)
    xy[points[:,:,-1] != 1] = (xy[points[...,-1] != 1].view(xy.shape[0],-1,2) - waypoints[:,j].unsqueeze(dim=1)).view(-1, 2)

    c, s = torch.cos(-heading), torch.sin(-heading)
    R = torch.stack([torch.stack([c,-s]), torch.stack([s,c])]).permute(2,1,0).to(xy.device)

    orientation = (heading + torch.atan2(*points[...,[3,2]].T)).T
    _points = torch.cat([
        torch.bmm(xy, R),
        torch.cos(orientation).unsqueeze(dim=-1),
        torch.sin(orientation).unsqueeze(dim=-1),
        points[...,4:] # velocity, class
    ], dim=-1)

    # ego-vehicle always facing forward
    _points[_points[:,:,-1]==1,[[2],[3]]] = torch.FloatTensor([[1.], [0.]]).to(xy.device)

    if waypoints_gt is not None:
        # NOTE: `waypoints` are predicted, we step through that and
        #       augment `waypoints_gt` to obtain "ground-truth"
        w = waypoints_gt - waypoints[:,[j]]
        # prune the ground-truth waypoints behind ego-vehicle after
        # stepping into the predicted waypoints
        mask = w[...,1]>0

        _waypoints_gt = torch.bmm(w, R)[mask] # over-computing but whatever...
        return _points, _waypoints_gt, mask

    _waypoints = torch.bmm(waypoints[:,j:]-waypoints[:,[j]], R)
    return _points, _waypoints


font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 16)
def visualize_birdview(points, action, waypoints, _waypoints=None, diameter=MAP_SIZE, r=0.5, w_r=0.5, **kwargs):
    canvas = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    canvas[...] = BACKGROUND
    canvas = Image.fromarray(canvas[...,:3])
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for i, (x, y) in enumerate(points[:,:2] + (diameter//2)):
        R = w_r if points[i,-1].item() == 0 else r
        draw.ellipse((x - R, diameter-y - R, x + R, diameter-y + R), fill=COLORS[int(points[i,-1].item())])

        if points[i,-1].item() != 0:
            heading = np.rad2deg(np.arctan2(*points[i,[3,2]])-(np.pi/2))
            draw.arc(xy=(x - 3, diameter-y - 3, x + 3, diameter-y + 3), start=heading-90, end=heading+90, width=1, fill=COLORS[int(points[i,-1].item())])

    for x, y in waypoints + (diameter//2):
        draw.ellipse((x - w_r, diameter-y - w_r, x + w_r, diameter-y + w_r), fill=(0, 175, 0))

    _w_r = kwargs.get('_w_r', w_r)
    if _waypoints is not None:
        for x, y in _waypoints + (diameter//2):
            draw.ellipse((x - _w_r, diameter-y - _w_r, x + _w_r, diameter-y + _w_r), fill=(175, 0, 0))

    draw.text((0, 0), ACTIONS[int(action)], fill='black', font=font)
    return canvas


if __name__ == '__main__':
    import sys
    import cv2

    data = PointDataset(sys.argv[1])

    i = 0; j = 0
    while True:
        _points, action, _waypoints = data[i]
        points, waypoints = step(_points, _waypoints, j)

        canvas = visualize_birdview(points, action, waypoints, r=1.0)
        cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        k = cv2.waitKey(0)
        if k == 100:   # d
            i = min(i+1,len(data)-1)
        elif k == 97: # a
            i = max(i-1,0)
        elif k == 113: # q
            break
        elif k >= 49 and k <= 53:
            j = k-49
        elif k == 119: # w
            j = min(j+1,4)
        elif k == 115: # s
            j = max(j-1,0)
        else:
            continue
