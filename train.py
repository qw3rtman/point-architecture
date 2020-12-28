import argparse
import time
import yaml
from collections import defaultdict

from pathlib import Path

import tqdm
import numpy as np
import torch
from torchvision.utils import make_grid

from .const import STEPS
from .model import AttentivePolicy
from .point_dataset import get_dataset, prune, step, visualize_birdview

import wandb


def log_visuals(points_batch, mask_batch, action_batch, waypoints_batch, _waypoints_batch, loss_batch, config):
    mask_batch = mask_batch.cpu()
    action_batch = action_batch.cpu()
    waypoints_batch = waypoints_batch.cpu()
    _waypoints_batch = _waypoints_batch.cpu()

    images = [(
        loss_batch[i].item(),
        torch.ByteTensor(np.uint8(visualize_birdview(points[mask_batch[i]], action_batch[i].item(), waypoints_batch[i], _waypoints_batch[i], r=1.0, diameter=config['data_args']['map_size'])).transpose(2,0,1))
    ) for i, points in enumerate(points_batch.cpu())]
    images.sort(key=lambda x: x[0], reverse=True)

    return make_grid([x[1] for x in images[:32]], nrow=4).numpy().transpose(1,2,0)


def train_or_eval(net, data, optim, is_train, config, bc=True, cf=False, od=False):
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
    for i, (points, mask, action, waypoints, steer, throttle, brake) in enumerate(iterator):
        points = points.to(config['device'])
        mask = mask.to(config['device'])
        action = action.to(config['device'])
        waypoints = waypoints.to(config['device'])
        steer = steer.to(config['device'])
        throttle = throttle.to(config['device'])
        brake = brake.to(config['device'])

        # vanilla behavior cloning
        if bc:
            _waypoints = net(points, mask, action)                   # pred traj at t
            bc_loss = criterion(_waypoints, waypoints)

        # forward consistency (t+1)
        if cf:
            if not bc:
                _waypoints = net(points, mask, action)           # pred traj at t
            # NOTE: if we detach _waypoints here, we treat the predicted waypoints as data
            _points, _waypoints_f = step(points, _waypoints, 1)  # pred map at t+1
            # NOTE: if we detach _points here, we skip the step, treat _points as data, not aug
            __waypoints = net(_points, mask, action)             # pred traj at t+1'
            # NOTE: if we detach _waypoints_f here, we basically augmented the map, treat it as
            #       data, and regress the policy onto the stepped predicted waypoints as data
            cf_loss = criterion(__waypoints[:,:STEPS-1], _waypoints_f)

        if od:
            if not bc and not cf:
                _waypoints = net(points, mask, action)           # pred traj at t
            # offline DAgger (t+1)
            # TODO: set 2*diameter in pad_collate before training
            # TODO: prune, feed into net, step on unpruned, prune, feed into net, etc.
            _points, waypoints_gt, _waypoints_gt, _waypoints_mask = step(points, _waypoints, 1, waypoints_gt=waypoints)
            __waypoints = net(_points, mask, action)
            od_loss = criterion(__waypoints[_waypoints_mask.flip(dims=(1,))], _waypoints_gt)

        # learned control
        # NOTE: by detaching, we don't bring this control signal into the policy. the policy
        #       shouldn't 1) adapt to cheat/trick the controller (by predicting easy waypoints
        #       or 2) learn bad control due to early (poor) waypoint predictions.
        speed = points[points[...,-1] == 1, [[4]]].T
        control_loss = net.control_loss(steer, throttle, brake,
                *net.control(_waypoints.detach(), speed).T)

        # combine all
        _loss = dict()
        if bc:
            _loss['bc'] = bc_loss.mean(dim=-1).mean(dim=-1)
        if cf:
            # NOTE: could weight the "correct" prediction higher than incorrect
            #       prediction (for picking ground-truth) via bc_loss
            _loss['cf'] = cf_loss.mean(dim=-1).mean(dim=-1) # simple mean
        if od:
            _loss['od'] = od_loss.mean(dim=-1).mean(dim=-1) # simple mean
        _loss['control'] = control_loss # NOTE: stop control early loss? overfits

        loss = torch.stack(list(_loss.values())).sum(dim=0) # per-sample loss
        loss_mean = loss.mean()

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())
        metrics = {'loss': loss_mean.item(),
                   'control_loss': control_loss.mean().item(),
                   'samples_per_second': points.shape[0] / (time.time() - tick)}
        if bc:
            metrics['bc_loss'] = bc_loss.mean().item()
        if cf:
            metrics['cf_loss'] = cf_loss.mean().item()
        if od:
            metrics['od_loss'] = od_loss.mean().item()

        if i == 0:
            metrics['images'] = [wandb.Image(log_visuals(points, mask, action, waypoints, _waypoints, loss, config))]
        wandb.log({('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses), {k: v.mean().item() for k, v in _loss.items()}


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

    _loss_val = dict()
    for epoch in tqdm.tqdm(range(wandb.run.summary['epoch']+1, parsed.max_epoch+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        bc, cf, od = get_schedule(config['schedule'], epoch, **_loss_val)
        loss_train, _ = train_or_eval(net, data_train, optim, True, config, bc, cf, od)
        with torch.no_grad():
            loss_val, _loss_val = train_or_eval(net, data_val, None, False, config, bc, cf, od)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})
        checkpoint_project(net, optim, scheduler, config)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


# TODO: find a good threshold
BC_LOSS_THRESHOLD = 0.25
def get_schedule(_id, epoch, **kwargs):
    bc_loss = kwargs.get('bc', float('inf'))
    if _id == 'bc':
        return True, False, False

    elif _id == 'bc-cf':
        return True, True, False
    elif _id == 'bc-add_cf':
        return True, bool(bc_loss < BC_LOSS_THRESHOLD), False
    elif _id == 'bc-switch_cf':
        return bool(bc_loss >= BC_LOSS_THRESHOLD), bool(bc_loss < BC_LOSS_THRESHOLD), False

    elif _id == 'bc-od':
        return True, False, True
    elif _id == 'bc-add_od':
        return True, False, bool(bc_loss < BC_LOSS_THRESHOLD)
    elif _id == 'bc-switch_od':
        return bool(bc_loss >= BC_LOSS_THRESHOLD), False, bool(bc_loss < BC_LOSS_THRESHOLD)

    elif _id == 'bc-add_od-add_cf':
        return True, bool(bc_loss < BC_LOSS_THRESHOLD), bool(bc_loss < BC_LOSS_THRESHOLD)
    elif _id == 'bc-switch_od-switch_cf':
        return bool(bc_loss >= BC_LOSS_THRESHOLD), bool(bc_loss < BC_LOSS_THRESHOLD), bool(bc_loss < BC_LOSS_THRESHOLD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--schedule', type=str, required=True)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Model args.
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)

    # Data args.
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--map_size', type=int, default=50)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    keys = ['batch_size', 'map_size', 'hidden_size', 'num_layers', 'num_heads', 'lr', 'weight_decay', 'schedule', 'description']
    run_name  = '_'.join(str(getattr(parsed, x)) for x in keys)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'schedule': parsed.schedule,
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
                'map_size': parsed.map_size
                },

            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay
                }
            }

    main(config, parsed)
