import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from .street_hazards import StreetHazardsClasses
from .misc import sliding_window_inference



class AccumulatorMIoU:
    """
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

    def compute(self):
        hist = self.confusion_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            iou_c = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou_c)
        return mean_iou
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


class AccumulatorBinaryAUPR:
    def __init__(self):
        self.reset()

    def add(self, preds, labels):
        if preds.dim() == 2: preds = preds.unsqueeze(0)
        if labels.dim() == 2: labels = labels.unsqueeze(0)
        preds, labels = preds.cpu(), labels.cpu()
        for p, l in zip(preds, labels):
            self.aupr_accumulator += average_precision_score(l.type(torch.int16).flatten(), p.type(torch.float16).flatten())
            self.auprs_count += 1

    def compute(self):
        return self.aupr_accumulator / self.auprs_count
    
    def reset(self):
        self.aupr_accumulator = 0
        self.auprs_count = 0


def evaluate_model(
        model,
        ds, 
        tot_classes = len(StreetHazardsClasses), 
        anomaly_class = StreetHazardsClasses.ANOMALY,
        batch_size = 1, 
        # patch_size = None, 
        compute_ap = True, 
        device = "cuda", 
    ):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    miou_acc = AccumulatorMIoU(tot_classes-1, anomaly_class)
    if compute_ap:
        aupr_acc = AccumulatorBinaryAUPR()

    for inputs, labels in (pbar := tqdm(dl)):
        inputs = inputs.to(device)

        with torch.no_grad():
            segm_pred, anom_pred = model(inputs)
            # if patch_size is None:
            #     segm_pred, anom_pred = model(inputs)
            # else:
            #     preds = []
            #     for image in inputs:
            #         preds.append( sliding_window_inference(model, image, patch_size, device).unsqueeze(0) )
            #     preds = torch.cat(preds)
            # if preds.dim() == 4: preds = preds.squeeze(1)
        segm_pred, anom_pred, labels = segm_pred.cpu(), anom_pred.cpu(), labels.cpu()
        segm_pred, anom_pred = segm_pred.squeeze(0), anom_pred.squeeze(0)

        miou_acc.add(torch.argmax(segm_pred, dim=0), labels)
        if compute_ap:
            aupr_acc.add(anom_pred, (labels == anomaly_class))

        pbar.set_description(f"mIoU: {miou_acc.compute()*100:.2f} -- AUPR: {aupr_acc.compute()*100 if compute_ap else 0:.2f}")

    return {
        "miou": miou_acc.compute(),
        "ap": aupr_acc.compute() if compute_ap else None
    }