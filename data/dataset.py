import os
import os.path as osp
from collections import OrderedDict
from glob import glob

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from utils.frame_utils import readFlow

from .augmentor import FlowAugmentor


def totensor(x):
    return torch.from_numpy(x).permute(2, 0, 1).float()


class CVO_sampler_lmdb:
    """Data sampling"""

    all_keys = ["imgs", "imgs_blur", "fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, is_training=True, keys=None):
        current_dir = osp.split(osp.realpath(__file__))[0]
        dst_dir = os.path.join(current_dir, "datasets", "CVO_full")
        if is_training:
            self.db_path = osp.join(dst_dir, "cvo_train.lmdb")
        else:
            self.db_path = osp.join(dst_dir, "cvo_test.lmdb")

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.samples = pa.deserialize(txn.get(b"__samples__"))
            self.length = len(self.samples)

        self.keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(self.keys)

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return self.length

    def sample(self, index):
        sample = OrderedDict()
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = "{:05d}_{:s}".format(index, k)
                value = pa.deserialize(txn.get(key.encode()))
                if "flow" in key:  # Convert Int to Floating
                    value = value.astype(np.float32)
                    value = (value - 2**15) / 128.0
                sample[k] = value
        return sample


class CVO(data.Dataset):
    all_keys = ["fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, keys=None, split="clean", is_training=True, crop_size=256):
        self.augmentor = FlowAugmentor(crop_size) if is_training else None

        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "clean":
            keys.append("imgs")
        else:
            keys.append("imgs_blur")

        self.sampler = CVO_sampler_lmdb(is_training, keys)

    def __getitem__(self, index):
        sample_dict = self.sampler.sample(index)
        if self.augmentor is not None:
            sample_dict = self.augmentor(sample_dict)

        out_dict = {}
        for k, v in sample_dict.items():
            v_ = totensor(np.ascontiguousarray(v).copy())
            if "imgs" in k:
                out_dict["imgs"] = v_
            else:
                out_dict[k] = v_

        return out_dict

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return len(self.sampler)


def fetch_train_dataloader(keys, batch=16, crop_size=256, split="clean", workers=0):
    """Create the data loader"""
    if "+" in split:
        dataset_clean = CVO(
            keys=keys,
            split="clean",
            is_training=True,
            crop_size=crop_size,
        )
        dataset_final = CVO(
            keys=keys,
            split="final",
            is_training=True,
            crop_size=crop_size,
        )
        dataset = dataset_clean + dataset_final
    else:
        dataset = CVO(
            keys=keys,
            split=split,
            is_training=True,
            crop_size=crop_size,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        pin_memory=True,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )
    return dataloader, dataset


def fetch_valid_dataloader(keys, split="clean", batch=1):
    if "+" in split:
        cleanpass = CVO(keys=keys, is_training=False, split="clean")
        finalpass = CVO(keys=keys, is_training=False, split="final")
        dataset = cleanpass + finalpass
    else:
        dataset = CVO(keys=keys, is_training=False, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataloader, dataset


class High_Speed_Sintel(data.Dataset):
    def __init__(self, data_dir, interv, blacklist=[]):
        super(High_Speed_Sintel, self).__init__()

        # set data dir
        self.data_dir = data_dir
        self.interv = interv
        self.sample_list = [
            osp.join(data_dir, x)
            for x in sorted(os.listdir(data_dir))
            if x not in blacklist
        ]

    def __getitem__(self, index):
        sample_dict = {}
        sintel_ori_path = osp.join(self.sample_list[index], "2_imgs")
        sintel_hs_path = osp.join(self.sample_list[index], "43_imgs")

        sintel_ori_list = sorted(glob(osp.join(sintel_ori_path, "*.png"))) + sorted(
            glob(osp.join(sintel_ori_path, "*.jpg"))
        )
        sintel_hs_list = sorted(glob(osp.join(sintel_hs_path, "*.png"))) + sorted(
            glob(osp.join(sintel_hs_path, "*.jpg"))
        )
        gt_flow = glob(osp.join(self.sample_list[index], "*.flo"))[0]
        occ_mask = glob(osp.join(self.sample_list[index], "*.png"))[0]

        # gt flow
        gt_flow = readFlow(gt_flow)
        gt_flow = totensor(gt_flow)
        sample_dict["gt_flow"] = gt_flow

        # occ mask
        occ_mask = totensor(cv2.imread(occ_mask)[..., 0:1])
        sample_dict["occ_mask"] = occ_mask / 255.0

        # original sintel images: 0~255,(C,436,1024)
        img1 = totensor(cv2.imread(sintel_ori_list[0])[..., ::-1].copy())
        img2 = totensor(cv2.imread(sintel_ori_list[1])[..., ::-1].copy())
        sample_dict["sintel_imgs"] = [img1, img2]

        # high-speed sintel images: 0~255,(C,436,1024)
        imgs_hs = []
        for i in range(0, len(sintel_hs_list), self.interv):
            imgs_hs.append(
                totensor(
                    cv2.resize(
                        cv2.imread(sintel_hs_list[i])[..., ::-1].copy(), (1024, 436)
                    )
                )
            )
        sample_dict["hs_sintel_imgs"] = imgs_hs  # list: img0,img7...img42

        return sample_dict

    def __len__(self):
        return len(self.sample_list)


def fetch_sintel_dataloader(data_root, interv=6, batch=10, blacklist=[]):
    """Create the data loader"""

    dataset = High_Speed_Sintel(data_root, interv, blacklist)
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return dataloader, dataset
