import torch.nn as nn
import math
import copy

from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import numpy as np
import torch

from .const import BACKGROUND, COLORS, ACTIONS

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 16)
def visualize_birdview(points, action, waypoints, _waypoints=None, r=0.5, w_r=0.5):
    canvas = np.zeros((80, 80, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)

    for x, y, c in points.flip((0,)):
        draw.ellipse((y+39 - r, 81-x - r, y+39 + r, 81-x + r), fill=COLORS[int(c.item())])

    for x, y in waypoints:
        draw.ellipse((y+39 - w_r, 60-x - w_r, y+39 + w_r, 60-x + w_r), fill=(0, 175, 0))

    if _waypoints is not None:
        for x, y in _waypoints:
            draw.ellipse((y+39 - w_r, 60-x - w_r, y+39 + w_r, 60-x + w_r), fill=(175, 0, 0))

    draw.text((0, 0), ACTIONS[action], fill='white', font=font)

    return canvas

def log_visuals(points_batch, mask_batch, action_batch, waypoints_batch, _waypoints_batch, loss_batch):
    images = [(
        loss_batch[i].mean().item(),
        torch.ByteTensor(np.uint8(visualize_birdview(points[mask_batch[i]], action_batch[i].item(), waypoints_batch[i], _waypoints_batch[i], w_r=1.5)).transpose(2,0,1))
    ) for i, points in enumerate(points_batch)]
    images.sort(key=lambda x: x[0], reverse=True)

    return make_grid([x[1] for x in images[:32]], nrow=4).numpy().transpose(1,2,0)
