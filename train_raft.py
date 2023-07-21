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
from loss import sequence_loss_raft
from utils import util
from utils.flow_viz import flow_to_image
from utils.util import Timer, count_parameters, get_timestamp, tbLogger


def set_default(args):
    args.resume = None
    """
    None,  not resume training;
    'auto', resume the latest state;
    Int, number of saving step.
    """
    if "debug" in args.exp_name.lower():
        args.valid_freq = 50
        args.log_freq = 1

    args.gamma = 0.85

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


def save_ckpt(epoch, step, scheduler, optimizer, model, args, latest=True):
    if latest:
        ckpt = "%s/latest.pth" % (args.ckpt_dir)
        state = "%s/latest.state" % (args.ckpt_dir)
    else:
        ckpt = "%s/%06d.pth" % (args.ckpt_dir, step)
        state = "%s/%06d.state" % (args.ckpt_dir, step)

    state_dict = {
        "epoch": epoch,
        "iter": step,
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(model.state_dict(), ckpt)
    torch.save(state_dict, state)


def send_message(message):
    headers = {
        "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjI5NDM5LCJ1dWlkIjoiOWJhNjA5MDYtOTkzMS00NGU5LTg5ODItOGRlYTJlZmI0ODFiIiwiaXNfYWRtaW4iOmZhbHNlLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.a_Xte9zi4NOzjyyDmzpsQFodW9rUOPU5ySHySR5CC8tvkCsl18aDyYkvD82I7QSfrGtA2IYxKonzlQMGZx8iWA"
    }
    resp = requests.post(
        "https://www.autodl.com/api/v1/wechat/message/send",
        json={
            "title": "AccFlow",
            "name": f"RAFT-CVO: {message}",
            "content": message,
        },
        headers=headers,
    )
    # print(resp.content.decode())


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
    tb_logger = tbLogger(args.log_dir)  # tensorboard

    ############# Build Data #############
    keys = ["fflows", "bflows", "delta_fflows", "delta_bflows"]
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
    send_message("Begin Training")
    logger.info(
        "Train on %d samples with batch %d, %d iters/epoch, %d iters in total",
        train_samples,
        args.batch,
        sample_per_epoch,
        num_steps,
    )

    ############# Build Model & Optimizer #############
    from networks.raft.raft import RAFT

    raft_args = argparse.Namespace(small=False, mixed_precision=True)
    model = RAFT(raft_args)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoints/raft-things.pth"))
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
        tb_logger.set_step(current_step)
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
            imgs = batch["imgs"]
            # randomly select one data pairs
            interval = np.random.randint(1, 7)
            direction = random.choice([-1, 1])
            if interval * direction == 1:  # local forward flow
                input = [imgs[0], imgs[1]]
                label = batch["delta_fflows"][0]
            elif interval * direction == -1:  # local backward flow
                input = [imgs[1], imgs[0]]
                label = batch["delta_bflows"][0]
            elif direction == 1:  # cross-frame forward flow
                input = [imgs[0], imgs[interval]]
                label = batch["fflows"][interval - 2]
            else:  # cross-frame backward flow
                input = [imgs[interval], imgs[0]]
                label = batch["bflows"][interval - 2]

            ############# add noise #############
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise = stdv * torch.randn(*input[0].shape, device=input[0].device)
                noise = 2 * (torch.clamp(noise, 0.0, 255.0) / 255.0) - 1
                input = [x + noise for x in input]

            ############# compute loss #############
            flows_pre = model(image1=input[0], image2=input[1], iters=12)
            loss, metrics = sequence_loss_raft(flows_pre, label, args.gamma)
            losses.append(loss.item())
            epes.append(metrics["epe"])

            ############# update params #############
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tb_logger.step()
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
                tb_logger.write_dict(
                    {"loss": avg_loss, "epe": avg_epe, "eta": eta_time}
                )

            ############# validation #############
            if current_step % args.valid_freq == 0 or current_step == num_steps - 1:
                logger.info("Evaluation Model %s" % args.exp_name)
                model.eval()
                metric_list = []
                val_result = {}
                for id, data in tqdm(enumerate(valid_loader), total=args.valid_sample):
                    data = preprocess(data)
                    input = data["imgs"]
                    label = data["bflows"]
                    with torch.no_grad():
                        _, FN0 = model(
                            image1=input[-1], image2=input[0], iters=20, test_mode=True
                        )
                    loss, metrics = sequence_loss_raft([FN0], label[-1], args.gamma)
                    metric_list.append(metrics)
                    val_result[id] = FN0
                    if id == args.valid_sample:
                        break
                avg_metric = {"val_" + k: [] for k in metric_list[0].keys()}
                for m in metric_list:
                    for k, v in m.items():
                        avg_metric["val_" + k].append(v)
                avg_metric = {k: sum(v) / len(v) for k, v in avg_metric.items()}
                save_ckpt(epoch, current_step, scheduler, optimizer, model, args, True)
                tb_logger.write_dict(avg_metric)  # XXX
                tb_logger.write_dict({"val_loss": loss})  # XXX
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
                    save_ckpt(
                        epoch, current_step, scheduler, optimizer, model, args, False
                    )
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
                send_message(f"Iter:{current_step:6,d}, loss:{loss:.3f}")

            model.train()

    tb_logger.close()
    send_message("Finish Training!")
    torch.save(model.state_dict(), "%s/final.pth" % args.ckpt_dir)
    logger.info("Finish training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="./configs/RAFT.yml")
    args = parser.parse_args()

    opt = util.parse_options(args.config)
    opt = EasyDict(opt)
    train(opt)
