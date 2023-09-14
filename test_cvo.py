import torch
import torch.nn as nn
from tqdm import tqdm

from data import dataset
from networks import build_flow_estimator
from networks.AccFlow_ import AccFlow
from networks.utils import backwarp


def build_acc_model(name, acc_ckpt):
    # name: acc+raft, acc+gma
    ofe = build_flow_estimator(name)
    ofe = ofe.cuda().eval()
    # state_dict = {k.replace("module.", ""): v for k, v in torch.load(ofe_ckpt).items()}
    # ofe.load_state_dict(state_dict)
    model = AccFlow(ofe)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(acc_ckpt))
    model.cuda().eval()
    return model


def build_ofe_model(name, ofe_ckpt):
    model = build_flow_estimator(name)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ofe_ckpt))
    model = model.cuda().eval()
    return model


def preprocess(batch):
    for key, value in batch.items():
        value = value.cuda()

        if "flow" in key:
            value = value.split(2, dim=1)
            assert len(value) in [5, 6], len(value)
        elif "imgs" in key:
            # imgs_org = value.split(3, dim=1)
            value = 2 * (value / 255.0) - 1
            value = value.split(3, dim=1)
            # value = [x[:, [2, 1, 0], ...] for x in value]  # RGB
            assert len(value) == 7, len(value)
        else:
            raise ValueError()

        batch[key] = value
    # batch["imgs_org"] = imgs_org
    return batch


def calc_occ_mask(bflow, fflow):
    """calculate occ mask in bidirection scheme
    input bflow, fflow:
        FN0 and F0N in (N,2,H,W)
    return occ_bw, occ_fw:
        binary map (N,1,H,W), 1 means occlusion, 0 means visible region.
    """
    occ_alpha_1 = 0.01
    occ_alpha_2 = 0.5

    def length_sq(x):
        temp = torch.sum(x**2, dim=1, keepdim=True)
        temp = torch.pow(temp, 0.5)
        return temp

    mag_sq = length_sq(fflow) + length_sq(bflow)
    flow_bw_warped = backwarp(bflow, fflow)
    flow_fw_warped = backwarp(fflow, bflow)
    flow_diff_fw = fflow + flow_bw_warped
    flow_diff_bw = bflow + flow_fw_warped
    occ_thresh = occ_alpha_1 * mag_sq + occ_alpha_2
    occ_fw = length_sq(flow_diff_fw) > occ_thresh
    occ_bw = length_sq(flow_diff_bw) > occ_thresh
    occ_fw = occ_fw.float()
    occ_bw = occ_bw.float()
    return occ_bw, occ_fw


def cal_epe(pred, label, occ_mask):
    """calculate epe based on mask
    pred: predicted flow
    label: gt flow
    occ_mask: occ mask, 1 means occ

    return
    epe_all: epe of all area
    epe_occ: epe of occluded area
    epe_vis: epe of non-occluded area
    """
    diff = torch.norm(pred - label, p=2, dim=1, keepdim=True)
    epe_all = torch.mean(diff, dim=(1, 2, 3))
    epe_occ = torch.sum(diff * occ_mask, dim=(1, 2, 3)) / torch.sum(
        occ_mask, dim=(1, 2, 3)
    )
    epe_vis = torch.sum((diff * (1 - occ_mask)), dim=(1, 2, 3)) / torch.sum(
        (1 - occ_mask), dim=(1, 2, 3)
    )

    return epe_all, epe_occ, epe_vis


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", type=str, choices=["clean", "final"])
parser.add_argument("--acc", "-acc", type=str, choices=["acc", "direct"])
parser.add_argument("--acc_ckpt", type=str, default=None)
parser.add_argument("--ofe", "-ofe", type=str, choices=["raft", "gma"])
parser.add_argument("--ofe_ckpt", type=str, default=None)
args = parser.parse_args()

batch_size = 10
torch.backends.cudnn.benchmark = True
end = 6  # CVO-6

model_name = args.acc + "|" + args.ofe
if "acc" in model_name:
    model = build_acc_model(model_name, args.acc_ckpt)
else:
    model = build_ofe_model(model_name, args.ofe_ckpt)

valid_loader, valid_dst = dataset.fetch_valid_dataloader(
    keys=["fflows", "bflows"], split=args.data, batch=batch_size
)

epe_all_list = []
epe_occ_list = []
epe_vis_list = []

for index, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
    data = preprocess(data)
    imgs = data["imgs"]
    # imgs_org = data["imgs_org"]
    bflos = data["bflows"]
    fflos = data["fflows"]

    imgs = imgs[: end + 1]  # im0, im1, im2...
    bflows = bflos[: end - 1]  # f20, f30, ...
    fflows = fflos[: end - 1]

    with torch.no_grad():
        if "acc" in model_name:
            FN0 = model(images=imgs, test_mode=False)[-1]
        else:
            FN0 = model(imgs[-1], imgs[0])

    bmask, fmask = calc_occ_mask(bflows[-1], fflows[-1])
    label = bflows[-1]
    epe_all, epe_occ, epe_vis = cal_epe(FN0, label, bmask)

    epe_all_list.append(epe_all)
    epe_occ_list.append(epe_occ)
    epe_vis_list.append(epe_vis)

avg_all = torch.mean(torch.cat(epe_all_list))
avg_vis = torch.mean(torch.cat(epe_vis_list))
avg_occ = torch.mean(torch.cat(epe_occ_list))

print("Finish".center(50, "="))
print("AVG EPE %s: " % model_name)
print("all:%.4f vis:%.4f occ:%.4f" % (avg_all, avg_vis, avg_occ))
with open("test_result_%s_E%d.txt" % (args.data, end), "a+") as f:
    f.write("AVG EPE %s: \n" % model_name)
    f.write("all:%.4f vis:%.4f occ:%.4f \n\n" % (avg_all, avg_vis, avg_occ))
