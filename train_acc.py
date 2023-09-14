import argparse
import logging
import os
import os.path as osp
import random

import numpy as np
import requests
import torch
import torch.nn as nn
from easydict import EasyDict
from PIL import Image
from torch import optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from data import dataset
from loss import sequence_loss_acc
from utils import util
from utils.flow_viz import flow_to_image
from utils.util import Timer, count_parameters, get_timestamp, tbLogger

MESSAGE_FREQ = 5000  # steps


def set_default(args):
    args.resume = None
    """
    None,  not resume training;
    'auto', resume the latest state;
    Int, number of saving step.
    """
    if "debug" in args.exp_name.lower():
        args.valid_freq = 10
        args.log_freq = 1

    args.log_dir = "./logs/%s" % args.exp_name
    args.ckpt_dir = "./checkpoints/%s" % args.exp_name
    if args.resume is None:  # rename if existed
        if osp.isdir(args.log_dir):
            os.rename(args.log_dir, args.log_dir + "_archived_" + get_timestamp())
        if osp.isdir(args.ckpt_dir):
            os.rename(args.ckpt_dir, args.ckpt_dir + "_archived_" + get_timestamp())
        os.makedirs(args.log_dir)
        os.makedirs(args.ckpt_dir)

    args.batch = args.batch_per_gpu * len(args.gpus)
    args.workers = args.batch

    return args


def preprocess(batch):
    # to cuda
    for key, value in batch.items():
        value = value.cuda()

        if "flow" in key:
            value = value.split(2, dim=1)
            assert len(value) in [5, 6], len(value)
        elif "imgs" in key:
            value = 2 * (value / 255.0) - 1
            value = value.split(3, dim=1)
            assert len(value) == 7, len(value)
        else:
            raise ValueError()

        batch[key] = value
    return batch


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


def save_flow(flow, path):
    # flow: N2HW
    flow = flow[0].cpu().permute(1, 2, 0).numpy()
    Image.fromarray(flow_to_image(flow)).save(path)


def save_ckpt(step, scheduler, optimizer, model, args, latest=True):
    if latest:
        ckpt = "%s/latest.pth" % (args.ckpt_dir)
        state = "%s/latest.state" % (args.ckpt_dir)
    else:
        ckpt = "%s/%06d.pth" % (args.ckpt_dir, step)
        state = "%s/%06d.state" % (args.ckpt_dir, step)

    state_dict = {
        "iter": step,
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(model.state_dict(), ckpt)
    torch.save(state_dict, state)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in args.gpus])
    args = set_default(args)

    ############# Build Logger #############
    util.setup_logger(
        "base",
        args.log_dir,
        "base_" + args.exp_name,
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")  # base logger
    # tb_logger = tbLogger(args.log_dir)  # tensorboard

    ############# Build Data #############
    keys = ["bflows"]
    train_loader, train_dst = dataset.fetch_train_dataloader(
        keys=keys,
        batch=args.batch,
        crop_size=args.image_size,
        split="clean+final",
        workers=args.workers,
    )
    valid_loader, valid_dst = dataset.fetch_valid_dataloader(
        keys=["bflows"], split="clean", batch=args.batch
    )
    train_samples = len(train_dst)
    sample_per_epoch = train_samples // args.batch + 1
    num_steps = sample_per_epoch * args.epochs
    args.num_steps = num_steps

    logger.info(
        "Train on %d samples with batch %d, %d iters/epoch, %d iters in total",
        train_samples,
        args.batch,
        sample_per_epoch,
        num_steps,
    )

    ############# Build Model & Optimizer #############
    from networks import build_flow_estimator
    from networks.AccFlow_ import AccFlow

    ofe = build_flow_estimator(args.exp_name)
    state_dict = {
        k.replace("module.", ""): v for k, v in torch.load(args.flow_pretrained).items()
    }
    ofe.load_state_dict(state_dict)
    for p in ofe.parameters():
        p.requires_grad = False
    model = AccFlow(ofe)
    model = nn.DataParallel(model)
    model.cuda()
    model.train()
    logger.info("model: %s" % args.exp_name)
    logger.info(
        "Parameter Count: trainable : %d, untrainble: %d" % count_parameters(model)
    )
    optimizer, scheduler = fetch_optimizer(args, model)
    if args.resume is not None:
        if args.resume.lower() == "auto":
            ckpt_resume = "%s/latest.pth" % args.ckpt_dir
            state_resume = "%s/latest.state" % args.ckpt_dir
        else:
            assert isinstance(args.resume, int), "Wrong resume value."
            ckpt_resume = "%s/%06d.pth" % (args.ckpt_dir, args.resume)
            state_resume = "%s/%06d.state" % (args.ckpt_dir, args.resume)
        ckpt = torch.load(ckpt_resume)
        state = torch.load(state_resume)
        logger.info("Loading ckpt & state from: \n%s \n%s", ckpt_resume, state_resume)
        model.load_state_dict(ckpt, strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        current_step = state["iter"]
        # tb_logger.set_step(current_step)
    else:
        current_step = 0

    ############# Build Loss #############
    logger.info("Loss: %s", args.loss_type.upper())

    ############# Misc #############
    scaler = GradScaler(enabled=args.mixed_precision)
    timer = Timer()

    start_epoch = current_step // sample_per_epoch
    logger.info("Start training from iter: {:d}".format(current_step))

    losses, epes = [], []
    best_val_epe = 1e10
    best_val_step = current_step
    for epoch in range(start_epoch, args.epochs):
        timer.tick()
        for _, batch in enumerate(train_loader):
            current_step += 1
            optimizer.zero_grad()
            batch = preprocess(batch)
            input = batch["imgs"]
            label = batch["bflows"]

            ############# add noise #############
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise = stdv * torch.randn(*input[0].shape, device=input[0].device)
                noise = 2 * (torch.clamp(noise, 0.0, 255.0) / 255.0) - 1
                input = [x + noise for x in input]

            ############# compute loss #############
            flows_pre = model(images=input, test_mode=False)
            loss, metrics = sequence_loss_acc(flows_pre, label)
            losses.append(loss.item())
            epes.append(metrics["epe"])

            ############# update params #############
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # tb_logger.step()
            timer.tick()

            ############# log #############
            if current_step % args.log_freq == 0 or current_step < 25:
                avg_time = timer.get_average_and_reset()
                eta_time = (avg_time * (num_steps - current_step)) / 3600
                avg_loss = sum(losses) / len(losses)
                avg_epe = sum(epes) / len(epes)
                logger.info(
                    f"<epoch:{epoch:2d}, iter:{current_step:6,d}, t:{avg_time:.2f}s, eta:{eta_time:.2f}h, loss:{avg_loss:.3f}, epe:{avg_epe:.3f}>"
                )
                losses, epes = [], []
                # tb_logger.write_dict(
                #     {"loss": avg_loss, "epe": avg_epe, "eta": eta_time}
                # )

            ############# validation #############
            if current_step % args.valid_freq == 0 or current_step == num_steps - 1:
                logger.info("Evaluation Model %s" % args.exp_name)
                model.eval()
                metric_list = []
                val_result = {}
                for id, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    data = preprocess(data)
                    input = data["imgs"]
                    label = data["bflows"]
                    with torch.no_grad():
                        out = model(images=input, test_mode=False)
                    loss, metrics = sequence_loss_acc(out, label)
                    metric_list.append(metrics)
                    val_result[id] = out[-1]

                avg_metric = {"val_" + k: [] for k in metric_list[0].keys()}
                for m in metric_list:
                    for k, v in m.items():
                        avg_metric["val_" + k].append(v)
                avg_metric = {k: sum(v) / len(v) for k, v in avg_metric.items()}
                save_ckpt(current_step, scheduler, optimizer, model, args, True)
                # tb_logger.write_dict(avg_metric)  # XXX
                # tb_logger.write_dict({"val_loss": loss})  # XXX
                epe = avg_metric["val_epe"]

                ############# if new best #############
                if epe <= best_val_epe:
                    best_val_epe = epe
                    best_val_step = current_step
                    ############# save samples #############
                    for index in args.visual_samples:
                        save_dir = osp.join(args.log_dir, "val/im%03d" % index)
                        os.makedirs(save_dir, exist_ok=True)
                        save_flow(
                            val_result[index],
                            osp.join(save_dir, "%06d.png" % (current_step)),
                        )
                    ############# save & clear checkpoints #############
                    save_ckpt(current_step, scheduler, optimizer, model, args, False)
                    ckpts = sorted(
                        [x for x in os.listdir(args.ckpt_dir) if ".pth" in x]
                    )
                    states = sorted(
                        [x for x in os.listdir(args.ckpt_dir) if ".state" in x]
                    )
                    assert len(ckpts) == len(states)
                    if len(ckpts) >= 4:
                        os.remove(osp.join(args.ckpt_dir, ckpts[0]))
                        os.remove(osp.join(args.ckpt_dir, states[0]))

                logger.info(
                    "Validation EPE: %.3f, current best EPE: %.3f(step: %s)"
                    % (epe, best_val_epe, best_val_step)
                )

            model.train()

    # tb_logger.close()
    torch.save(model.state_dict(), "%s/final.pth" % args.ckpt_dir)
    logger.info("Finish training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="./configs/RAFT.yml")
    args = parser.parse_args()

    opt = util.parse_options(args.config)
    opt = EasyDict(opt)
    train(opt)
