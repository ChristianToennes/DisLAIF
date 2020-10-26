import numpy as np

def acc(y_true, y_pred):
    return np.max(np.count_nonzero(y_true==y_pred, axis=tuple(range(1,len(y_true.shape))) ))/np.size(y_true[0])

def precision(y_true, y_pred):
    if (y_true.shape[-1] > 1):
        tp = np.count_nonzero(y_true[...,1]*y_pred[...,1])
        fp = np.count_nonzero(y_pred[...,1]*(1-y_true[...,1]))
    else:
        tp = np.count_nonzero(y_true*y_pred)
        fp = np.count_nonzero(y_pred*(1-y_true))
    if (tp+fp)==0:
        return 0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    if (y_true.shape[-1] > 1):
        tp = np.count_nonzero(y_true[...,1]*y_pred[...,1])
        total = np.count_nonzero(y_true[...,1])
    else:
        tp = np.count_nonzero(y_true*y_pred)
        total = np.count_nonzero(y_true)
    return tp / total

def label_acc(y_true, y_pred):
    a = np.bitwise_and(y_true,y_pred)
    nom = max(np.count_nonzero(a, axis=tuple(range(1,len(y_true.shape))) ))
    denom = np.count_nonzero(y_true)
    if denom == 0:
        return 0
    return nom / denom
