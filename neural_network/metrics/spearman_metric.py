import numpy as np
import torch
from scipy.stats import spearmanr
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort

device = "cuda" if torch.cuda.is_available() else "cpu"


class SpearmanCorrCoefMetric:
    def __init__(self):
        self.value = 0.0
        self.n_batch = 0

    def update(self, pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        self.value += spearmanr(pred, np.swapaxes(target, 0, 1))[0]*-1

        self.n_batch += 1

    def evaluate(self):
        if self.n_batch > 0:
            score = self.value / self.n_batch
        else:
            score = -float('inf')

        return score


def corrcoef(pred, target):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()

    return (pred_n * target_n).sum()


def differentiable_spearman(pred, target, regularization="l2", regularization_strength=1e-2):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank

    target = target.cpu()
    pred = soft_rank(torch.unsqueeze(pred, dim=0).cpu(),
                     regularization=regularization,
                     regularization_strength=regularization_strength)

    return corrcoef(pred / pred.shape[-1], target)

