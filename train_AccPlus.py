from utils import util
from utils.util import Timer
from torch.utils.tensorboard import SummaryWriter
from data import datasets
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
from easydict import EasyDict
import logging
from torch.cuda.amp import GradScaler
from PIL import Image
from tqdm import tqdm
from utils.flow_viz import flow_to_image
from datetime import datetime


def set_default(args):
    args.name = 'AccPlus'
    args.resume = 'auto'
    '''
    None,  not resume training;
    'auto', resume the latest state;
    Int, number of saving step.
    '''

    if 'debug' in args.name.lower():
        args.valid_freq = 50
        args.log_freq = 1
    args.name += '_%s' % args.model_id
    args.sampler = 'lmdb'

    args.data_keys = dict(flo='bflows')

    return args


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def preprocess(batch):
    # to cuda
    for key, value in batch.items():
        value = value.cuda()

        if 'flow' in key:
            value = value.split(2, dim=1)
            assert len(value) in [5, 6], len(value)
        elif 'imgs' in key:
            value = value.split(3, dim=1)
            assert len(value) == 7, len(value)
        else:
            raise ValueError()

        batch[key] = value
    return batch


def sequence_loss(flow_pred, flow_gt):
    """ Loss function defined over sequence of flow predictions """
    if isinstance(flow_gt, (list, tuple)):
        assert len(flow_pred) == len(flow_gt), 'length not match!'
        loss_seq = 0
        for i in range(len(flow_pred)):
            loss = (flow_pred[i] - flow_gt[i]).abs()
            loss_seq += loss.mean()
        epe = torch.sum((flow_pred[-1] - flow_gt[-1])**2, dim=1).sqrt()
        epe = epe.view(-1)
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return loss_seq, metrics

    elif isinstance(flow_gt, torch.Tensor):

        loss = (flow_pred - flow_gt).abs()
        epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        return loss, metrics

    else:
        raise NotImplementedError('not implemented..')


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = sum(p.numel() for p in model.parameters()
                      if not p.requires_grad)
    return trainable, untrainable


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.wdecay,
                            eps=args.epsilon)

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        args.lr,
                                        args.num_steps + 100,
                                        pct_start=0.05,
                                        cycle_momentum=False,
                                        anneal_strategy='linear')

    return optimizer, scheduler


def resume_optimizer(resume_state, optimizer, scheduler):
    '''Resume the optimizers and schedulers for training'''
    resume_optimizer = resume_state['optimizer']
    resume_scheduler = resume_state['scheduler']

    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return optimizer, scheduler


def save_flow(flow, path):
    # flow: N2HW
    flow = flow[0].cpu().permute(1, 2, 0).numpy()
    Image.fromarray(flow_to_image(flow)).save(path)


class tbLogger:

    def __init__(self, log_dir):
        self.total_steps = 0
        self.writer = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.total_steps += 1

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

    def set_step(self, step):
        self.total_steps = step


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in args.gpus])
    args = set_default(args)

    ############# Build Logger #############
    log_dir = './logs/%s' % args.name
    ckpt_dir = './checkpoints/%s' % args.name
    if not args.resume:
        if os.path.isdir(log_dir):  # rename if existed
            os.rename(log_dir, log_dir + '_archived_' + get_timestamp())
        if os.path.isdir(ckpt_dir):
            os.rename(ckpt_dir, ckpt_dir + '_archived_' + get_timestamp())
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    args.log_dir = log_dir

    util.setup_logger('base',
                      './logs/%s' % args.name,
                      'base_' + args.name,
                      level=logging.INFO,
                      screen=True,
                      tofile=True)
    logger = logging.getLogger('base')  # base logger
    tb_logger = tbLogger(log_dir)  # tensorboard

    ############# Build Data #############
    keys = list(args.data_keys.values())
    train_loader, train_dst = datasets.fetch_train_dataloader(
        args.train_sample,
        keys,
        args.batch_size,
        args.image_size,
    )
    valid_loader, valid_dst = datasets.fetch_valid_dataloader(
        args.valid_sample, keys, split='clean+final')
    train_samples = len(train_dst)
    sample_per_epoch = train_samples // args.batch_size + 1
    num_steps = sample_per_epoch * args.epochs
    args.num_steps = num_steps
    logger.info(
        'Train on %d samples with batch %d, %d iters/epoch, %d iters in total',
        train_samples, args.batch_size, sample_per_epoch, num_steps)

    ############# Build Model #############
    from networks.AccPlus import AccPlus
    model = AccPlus(ofe=args.model_id)
    model = nn.DataParallel(model)
    model.cuda()
    model.train()
    logger.info("model: %s" % args.name)
    logger.info("Parameter Count: trainable : %d, untrainble: %d" %
                count_parameters(model))
    optimizer, scheduler = fetch_optimizer(args, model)
    if args.resume is not None:
        if args.resume.lower() == 'auto':
            ckpt_resume = './checkpoints/%s/latest.pth' % args.name
            state_resume = './checkpoints/%s/latest.state' % args.name

            ckpt = torch.load(ckpt_resume)
            state = torch.load(state_resume)
        else:
            assert isinstance(args.resume, int)
            ckpt_resume = './checkpoints/%s/%06d.pth' % (args.name,
                                                         args.resume)
            state_resume = './checkpoints/%s/%06d.state' % (args.name,
                                                            args.resume)
            ckpt = torch.load(ckpt_resume)
            state = torch.load(state_resume)
        logger.info('Loading ckpt & state from: \n%s \n%s', ckpt_resume,
                    state_resume)
        model.load_state_dict(ckpt, strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        current_step = state['iter']
        tb_logger.set_step(current_step)
    else:
        current_step = 0

    ############# Build Loss #############
    logger.info("Loss: %s", args.loss_type.upper())

    ############# Misc #############
    scaler = GradScaler(enabled=args.mixed_precision)
    timer = Timer()

    start_epoch = current_step // sample_per_epoch
    logger.info('Start training from iter: {:d}'.format(current_step))
    losses = []
    best_val_epe = 1e10  # 用来记录当前最好的eval epe, 在训练过程中print出来
    best_val_step = 0

    for epoch in range(start_epoch, args.epochs):
        for _, batch in enumerate(train_loader):
            current_step += 1
            optimizer.zero_grad()
            batch = preprocess(batch)
            imgs = batch['imgs']
            flos = batch[args.data_keys['flo']]
            ############# add noise #############
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise = stdv * torch.randn(*imgs[0].shape,
                                           device=imgs[0].device)
                noise = torch.clamp(noise, 0., 255.)
                imgs = [x + noise for x in imgs]
            ############# compute loss #############

            flows_pre = model(images=imgs, iters=12, test_mode=False)
            loss, metrics = sequence_loss(flows_pre, flos)
            losses.append(loss.item())

            ############# update params #############
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            tb_logger.step()

            ############# log #############
            if current_step % args.log_freq == 0 or current_step < 10:

                def eta(t_iter):
                    return (t_iter * (num_steps - current_step)) / 3600

                avg_time = timer.get_average_and_reset()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.2f}s, eta:{:.2f}h, {:s}_loss:{:.3f}, epe:{:.3f}> '.format(
                    epoch, current_step,
                    scheduler.get_last_lr()[0], avg_time, eta(avg_time),
                    args.loss_type,
                    sum(losses) / len(losses), metrics['epe'])
                logger.info(message)
                losses = []
            timer.tick()

            ############# validation #############
            if current_step % args.valid_freq == 0 or current_step == num_steps - 1:
                logger.info('Evaluation Model %s' % args.name)
                model.eval()
                epe_list = []
                val_result = {}
                for index, val_data in tqdm(enumerate(valid_loader),
                                            total=len(valid_dst)):
                    val_data = preprocess(val_data)
                    imgs = val_data['imgs']
                    flos = val_data[args.data_keys['flo']]

                    with torch.no_grad():
                        flow_pre = model(images=imgs, iters=20, test_mode=True)
                    FN0 = flow_pre[-1]
                    epe = torch.sum((FN0 - flos[-1])**2, dim=1).sqrt().view(-1)
                    epe_list.append(epe)

                    val_result[index] = FN0
                epe_list = torch.cat(epe_list)
                epe = epe.mean().item()
                l_1 = (epe_list < 1).float().mean().item()
                l_3 = (epe_list < 3).float().mean().item()
                l_5 = (epe_list < 5).float().mean().item()
                ckpt = './checkpoints/%s/latest.pth' % (args.name)
                state = './checkpoints/%s/latest.state' % (args.name)
                state_dict = {
                    'epoch': epoch,
                    'iter': current_step,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(model.state_dict(), ckpt)
                torch.save(state_dict, state)
                ############# if new best #############
                if epe <= best_val_epe:
                    best_val_epe = epe
                    best_val_step = current_step
                    ############# save samples #############
                    for index in args.visual_samples:
                        save_dir = os.path.join(log_dir, 'val/im%03d' % index)
                        os.makedirs(save_dir, exist_ok=True)
                        save_flow(
                            val_result[index],
                            os.path.join(save_dir,
                                         '%06d.png' % (current_step + 1)))
                    ############# save ckpt & state #############
                    ckpt = './checkpoints/%s/%06d.pth' % (args.name,
                                                          current_step)
                    state = './checkpoints/%s/%06d.state' % (args.name,
                                                             current_step)
                    state_dict = {
                        'epoch': epoch,
                        'iter': current_step,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(model.state_dict(), ckpt)
                    torch.save(state_dict, state)
                    ckpts = sorted(
                        [x for x in os.listdir(ckpt_dir) if '.pth' in x])
                    states = sorted(
                        [x for x in os.listdir(ckpt_dir) if '.state' in x])
                    assert len(ckpts) == len(states)
                    if len(ckpts) >= 4:
                        os.remove(os.path.join(ckpt_dir, ckpts[0]))
                        os.remove(os.path.join(ckpt_dir, states[0]))

                ############# to tensorboard #############
                tb_logger.write_dict({
                    'EPE': epe,
                    '<1px': l_1,
                    '<3px': l_3,
                    '<5px': l_5,
                    'eta': eta(avg_time)
                })
                logger.info(
                    'Validation EPE: %.3f, current best EPE: %.3f(step: %s)' %
                    (epe, best_val_epe, best_val_step))

            model.train()

    tb_logger.close()
    pth = 'checkpoints/%s/checkpoint.pth' % args.name
    torch.save(model.state_dict(), pth)

    logger.info("Finish training, save to %s", pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=2)
    parser.add_argument(
        '--model_id',
        '-m',
        type=str,
        default='raft-things',
        choices=['raft-things', 'raft-kubric', 'gma-things', 'gma-kubric'])
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default="./configs/AccFlow.yml")
    args = parser.parse_args()

    opt = util.parse_options(args.config)
    opt = EasyDict(opt)
    opt.model_id = args.model_id
    opt.batch_size = args.batch
    train(opt)
