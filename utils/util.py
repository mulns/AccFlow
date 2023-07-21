import logging
import os
import time
from collections import OrderedDict
from datetime import datetime

import yaml
from torch.utils.tensorboard import SummaryWriter


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg


def parse_options(root_path):
    # parse yml to dict
    with open(root_path, mode="r") as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    return opt


def get_timestamp():
    return datetime.now().strftime("%y|%m|%d-%H:%M:%S")


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=True, tofile=True
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, untrainable


class ScopeTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{} {:.3E}".format(self.name, self.interval))


class Timer:
    def __init__(self):
        self.times = []

    def tick(self):
        self.times.append(time.time())

    def get_average_and_reset(self):
        if len(self.times) < 2:
            return -1
        avg = (self.times[-1] - self.times[0]) / (len(self.times) - 1)
        self.times = [self.times[-1]]
        return avg

    def get_last_iteration(self):
        if len(self.times) < 2:
            return 0
        return self.times[-1] - self.times[-2]


class TickTock:
    def __init__(self):
        self.time_pairs = []
        self.current_time = None

    def tick(self):
        self.current_time = time.time()

    def tock(self):
        assert self.current_time is not None, self.current_time
        self.time_pairs.append([self.current_time, time.time()])
        self.current_time = None

    def get_average_and_reset(self):
        if len(self.time_pairs) == 0:
            return -1
        deltas = [t2 - t1 for t1, t2 in self.time_pairs]
        avg = sum(deltas) / len(deltas)
        self.time_pairs = []
        return avg

    def get_last_iteration(self):
        if len(self.time_pairs) == 0:
            return -1
        return self.time_pairs[-1][1] - self.time_pairs[-1][0]


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
