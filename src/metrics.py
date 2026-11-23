"""Accuracy, confusion matrix and per-class accuracy written by us (no sklearn)."""

import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes=6):
    """Rows = true activity, columns = predicted. C[j, k] = count of true j, pred k."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C


def per_class_accuracy(C):
    """Cjj / row sum, i.e. how many of activity j were recognised as j."""
    row = C.sum(axis=1).astype(float)
    out = np.zeros(C.shape[0], dtype=float)
    for j in range(C.shape[0]):
        if row[j] > 0:
            out[j] = C[j, j] / row[j]
    return out
