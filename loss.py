import torch


def sequence_loss_raft(flow_preds, flow_gt, gamma):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_acc(flow_preds, flow_gts):
    """Loss function defined over sequence of flow predictions"""
    assert len(flow_preds) == len(flow_gts), "length not match!"
    loss_seq = 0
    for i in range(len(flow_preds)):
        loss = (flow_preds[i] - flow_gts[i]).abs()
        loss_seq += loss.mean()
    epe = torch.sum((flow_preds[-1] - flow_gts[-1]) ** 2, dim=1).sqrt()
    epe = epe.view(-1)
    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return loss_seq, metrics
