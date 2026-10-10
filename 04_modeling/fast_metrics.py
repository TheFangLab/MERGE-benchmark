"""AUROC and average precision of a fixed score vector under integer sample weights.

A bootstrap resample is a vector of per-variant counts. Evaluating the metrics
from those counts on a presorted score vector gives the same values as
sklearn's roc_auc_score / average_precision_score on the resampled rows (to
floating-point rounding, checked by `check_against_sklearn`), without
re-sorting for every resample.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class WeightedScorer:
    """AUROC and average precision under integer sample weights."""

    def __init__(self, score, y):
        score = np.asarray(score, float)
        self.order = np.argsort(-score, kind="mergesort")
        s = score[self.order]
        self.group = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        self.n_groups = int(self.group[-1]) + 1
        self.y = np.asarray(y, int)[self.order]

    def __call__(self, w):
        w = np.asarray(w, float)[self.order]
        tp = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        fp = np.bincount(self.group, weights=w * (1 - self.y), minlength=self.n_groups)
        keep = (tp + fp) > 0                     # thresholds present in the resample
        ctp, cfp = np.cumsum(tp[keep]), np.cumsum(fp[keep])
        tpr = np.concatenate([[0.0], ctp / ctp[-1]])
        fpr = np.concatenate([[0.0], cfp / cfp[-1]])
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
        ap = float(np.sum(np.diff(tpr) * (ctp / (ctp + cfp))))
        return auroc, ap


def check_against_sklearn(y, p, seed=0, n=3):
    """Assert agreement with sklearn on a few random resamples."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    rng = np.random.RandomState(seed)
    sc = WeightedScorer(p, y)
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        a, b = sc(np.bincount(idx, minlength=len(y)))
        assert abs(a - roc_auc_score(y[idx], p[idx])) < 1e-12
        assert abs(b - average_precision_score(y[idx], p[idx])) < 1e-12
