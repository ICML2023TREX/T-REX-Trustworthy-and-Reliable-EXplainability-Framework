import numpy as np
import torch


def iou_f1_precision_recall(attn_true: torch.Tensor, attn_pred: torch.Tensor):
    tp = ((attn_true == 1) & attn_pred).sum().item()
    fp = ((attn_true == 0) & attn_pred).sum().item()
    fn = ((attn_true == 1) & ~attn_pred).sum().item()

    # take into account only factual explanations
    union = ((attn_true == 1) | attn_pred).sum().item()
    iou = tp / (union + 1e-8)

    # take into account factual and counterfactual explanations
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return iou, f1, precision, recall


def mae_maeFP_maeFN(attn_true: torch.Tensor, attn_pred: torch.Tensor):
    mae = torch.abs(attn_true - attn_pred).sum().item() / \
          (torch.ones(attn_true.size()).sum().item() + 1e-8)
    maeFP = torch.where(attn_true > 0, 0, attn_pred).sum().item() / \
        (torch.where(attn_true > 0, 0, 1).sum().item() + 1e-8)
    maeFN = torch.abs(torch.where(attn_true > 0, attn_pred, 0) - attn_true).sum().item() / \
        (torch.where(attn_true > 0, 1, 0).sum().item() + 1e-8)
    return mae, maeFP, maeFN


class AverageMeter:
    def __init__(self):
        self._metrics = None
        self.reset()

    def reset(self):
        self._metrics = []

    def update(self, value: float):
        self._metrics.append(value)

    def compute(self, reset: bool = True):
        result = np.mean(self._metrics)
        if reset:
            self.reset()
        return result

