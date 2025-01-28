import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

from .street_hazards import StreetHazardsClasses
from .predictor import BasePredictor


class AccumulatorMIoU:
    """
    Accumulates a confusion matrix and computes mIoU when needed.
    Adapted from https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py
    """
    def __init__(self, num_classes, anomaly_index):
        self.num_classes = num_classes
        self.anomaly_index = anomaly_index if anomaly_index is not None else -1
        self.reset()

    def _fast_hist(self, preds, labels):
        mask = (labels >= 0) & (labels != self.anomaly_index)
        hist = np.bincount(
            self.num_classes * labels[mask].type(torch.int32) + preds[mask],
            minlength = self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist
    
    def add(self, preds, labels):
        if preds.dim() == 2: preds = preds.unsqueeze(0)
        if labels.dim() == 2: labels = labels.unsqueeze(0)
        preds, labels = preds.cpu(), labels.cpu()
        for p, l in zip(preds, labels):
            self.confusion_matrix += self._fast_hist( p.flatten(), l.flatten() )

    def compute(self, return_classwise_iou=False):
        hist = self.confusion_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            iou_c = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou_c)
        return (mean_iou, iou_c.tolist()) if return_classwise_iou else mean_iou
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


class AccumulatorAUPR:
    def __init__(self):
        """
        Accumulates AUPRs and averages them when needed.
        """
        self.reset()

    def add(self, preds, labels):
        if preds.dim() == 2: preds = preds.unsqueeze(0)
        if labels.dim() == 2: labels = labels.unsqueeze(0)
        preds, labels = preds.cpu(), labels.cpu()
        for p, l in zip(preds, labels):
            self.aupr_accumulator += average_precision_score(l.type(torch.int32).flatten().numpy(), p.type(torch.float32).flatten().numpy())
            self.auprs_count += 1

    def compute(self):
        return self.aupr_accumulator / self.auprs_count
    
    def reset(self):
        self.aupr_accumulator = 0
        self.auprs_count = 0


@torch.no_grad()
def evaluate_model(
        predictor: BasePredictor,
        ds: Dataset, 
        tot_classes = len(StreetHazardsClasses), 
        anomaly_class = StreetHazardsClasses.ANOMALY,
        batch_size = 1, 
        compute_miou = True, 
        compute_ap = True, 
        device = "cuda", 
    ):
    """
    Evaluates a predictor on a dataset.
    """
    assert compute_miou or compute_ap, "Requested to compute nothing"
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    predictor.to(device)
    miou_acc = AccumulatorMIoU(tot_classes-1, anomaly_class) if compute_miou else None
    aupr_acc = AccumulatorAUPR() if compute_ap else None

    for inputs, labels in (pbar := tqdm(dl)):
        inputs = inputs.to(device)
        labels = labels.cpu()

        segm_maps, anom_preds = predictor(inputs)
        segm_maps, anom_preds = segm_maps.cpu(), anom_preds.cpu()

        for i in range(len(inputs)):
            if compute_miou:
                miou_acc.add(segm_maps[i], labels[i])
            if compute_ap:
                aupr_acc.add(anom_preds[i], (labels[i] == anomaly_class))

        pbar.set_description(f"mIoU: {miou_acc.compute()*100 if compute_miou else 0:.2f} -- AUPR: {aupr_acc.compute()*100 if compute_ap else 0:.2f}")

    out = {}
    if compute_miou:
        miou, iou_c = miou_acc.compute(return_classwise_iou=True)
        out["miou"] = miou
        out["iou_c"] = iou_c
    if compute_ap:
        out["ap"] = aupr_acc.compute()
    return out