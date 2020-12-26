from .model import AttentivePolicy
from .point_dataset import get_episode, step, prune, visualize_birdview
import torch
import numpy as np


def nothing(data, map_size):
    cv2.namedWindow('map', cv2.WINDOW_NORMAL)

    i = 0
    while True:
        points, action, waypoints, _, _, _ = data[i]

        canvas = visualize_birdview(points, action, waypoints, r=1.0, w_r=0.5, diameter=map_size)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        k = cv2.waitKey(0)
        if k == 100:
            i = min(i+1,len(data)-1)
        elif k == 97: #108: # l
            i = max(i-1,0)
        elif k == 113: # q
            break


def drive_around(net, data, map_size):
    points, action, waypoints, _, _, _ = data[0]
    #points = prune(points, 2*map_size)
    points = points.reshape(1,-1,6).cuda()
    mask = torch.ones((1, points.shape[0]), dtype=torch.bool).cuda()
    action = action.item()
    waypoints = waypoints.cuda()

    cv2.namedWindow('map', cv2.WINDOW_NORMAL)

    k = -1
    while True:
        action = torch.LongTensor([action]).cuda()
        _waypoints = net(points, mask, action).detach()
        if k == 32:
            points, waypoints = step(points, _waypoints, 1) # step thru pred

        canvas = visualize_birdview(points.cpu(), action, waypoints.cpu() if k == 32 else _waypoints.cpu(), r=1.0, w_r=0.5, diameter=map_size)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        k = cv2.waitKey(0)
        if k == 100: # 114: # r
            action = 1
        elif k == 97: #108: # l
            action = 0
        elif k == 119: #102: # f
            action = 3
        elif k == 115: # s
            action = 2
        elif k == 113: # q
            break


def forward_consistency(net, data, map_size):
    cv2.namedWindow('map', cv2.WINDOW_NORMAL)

    i = 0
    while True:
        points, action, waypoints, _, _, _ = data[i]
        #points = prune(points, 2*map_size)
        points = points.reshape(1,-1,6).cuda()
        mask = torch.ones((1, points.shape[0]), dtype=torch.bool).cuda()
        action = action.reshape(1).cuda()
        waypoints = waypoints.cuda()

        _waypoints = net(points, mask, action)
        _points, _waypoints = step(points, _waypoints, 1)
        __waypoints = net(_points, mask, action)

        canvas = visualize_birdview(points.cpu(), action, _waypoints.cpu(), __waypoints.cpu(), r=1.0, w_r=0.5, _w_r=0.1, diameter=map_size)
        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('map', 400, 400)

        k = cv2.waitKey(0)
        if k == 100:
            i = min(i+1,len(data)-1)
        elif k == 97: #108: # l
            i = max(i-1,0)
        elif k == 113: # q
            break


if __name__ == '__main__':
    import sys
    import cv2
    import yaml
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=Path, required=True)
    parser.add_argument('--policy', '-p', type=Path, required=True)
    args = parser.parse_args()

    data = get_episode(str(args.dataset))

    config = yaml.load((args.policy.parent/'config.yaml').read_text())
    map_size = config['data_args']['value']['map_size']
    net = AttentivePolicy(**config['model_args']['value'])
    net.load_state_dict(torch.load(args.policy))
    net.cuda()
    net.eval()

    #nothing(data)
    #drive_around(net, data, map_size)
    forward_consistency(net, data, map_size)
