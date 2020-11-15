import argparse
import time
import yaml
from collections import defaultdict

from pathlib import Path

import tqdm
import numpy as np
import torch
from torchvision.utils import make_grid

from .model import AttentivePolicy
from .point_dataset import get_dataset, visualize_birdview

import wandb


def log_visuals(points_batch, mask_batch, action_batch, waypoints_batch, _waypoints_batch, loss_batch):
    mask_batch = mask_batch.cpu()
    action_batch = action_batch.cpu()
    waypoints_batch = waypoints_batch.cpu()
    _waypoints_batch = _waypoints_batch.cpu()

    images = [(
        loss_batch[i].mean().item(),
        torch.ByteTensor(np.uint8(visualize_birdview(points[mask_batch[i]], action_batch[i].item(), waypoints_batch[i], _waypoints_batch[i], r=1.0, w_r=2.0)).transpose(2,0,1))
    ) for i, points in enumerate(points_batch.cpu())]
    images.sort(key=lambda x: x[0], reverse=True)

    return make_grid([x[1] for x in images[:32]], nrow=4).numpy().transpose(1,2,0)


def train_or_eval(net, data, optim, is_train, config):
    if is_train:
        desc = 'train'
        net.train()
    else:
        desc = 'val'
        net.eval()

    losses = list()
    criterion = torch.nn.L1Loss(reduction='none')

    tick = time.time()
    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)
    for i, (M_pad, M_mask, action, waypoints) in enumerate(iterator):
        M_pad = M_pad.to(config['device'])
        M_mask = M_mask.to(config['device'])
        action = action.to(config['device'])
        waypoints = waypoints.to(config['device'])

        _waypoints = net(M_pad, M_mask, action)
        loss = criterion(_waypoints, waypoints)
        loss_mean = loss.mean()

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())
        metrics = {'loss': loss_mean.item(),
                   'samples_per_second': M_pad.shape[0] / (time.time() - tick)}
        if i == 0:
            metrics['images'] = [wandb.Image(log_visuals(M_pad, M_mask, action, waypoints, _waypoints, loss))]
        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config, parsed):
    net = AttentivePolicy(**config['model_args']).to(config['device'])

    data_train, data_val = get_dataset(**config['data_args'])
    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5,
            milestones=[mult * config['max_epoch'] for mult in [0.5, 0.75]])

    project_name = 'attentive-policy'
    wandb.init(project=project_name, config=config, name=config['run_name'],
            resume=True, id=str(hash(config['run_name'])))
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0

    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, parsed.max_epoch+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        loss_train = train_or_eval(net, data_train, optim, True, config)
        with torch.no_grad():
            loss_val = train_or_eval(net, data_val, None, False, config)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})
        checkpoint_project(net, optim, scheduler, config)
        if epoch % 50 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Model args.
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['hidden_size', 'lr', 'weight_decay', 'batch_size', 'description']
    run_name  = '_'.join(str(getattr(parsed, x)) for x in keys)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'model_args': {
                'hidden_size': parsed.hidden_size,
                'num_layers': parsed.num_layers,
                'nhead': parsed.num_heads
                },

            'data_args': {
                'num_workers': 8,
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config, parsed)
