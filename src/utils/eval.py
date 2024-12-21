import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.classification import binary_average_precision

from .street_hazards import StreetHazardsClasses
from .misc import sliding_window_inference


def evaluate_model(model, ds, device, batch_size=1, patch_size=None, compute_ap=True, num_classes=13):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    mious = []
    aps = []

    for inputs, labels in tqdm(dl):
        inputs = inputs.to(device)

        with torch.no_grad():
            if patch_size is None:
                preds = model(inputs)
            else:
                preds = []
                for image in inputs:
                    preds.append( sliding_window_inference(model, image, patch_size, device).unsqueeze(0) )
                preds = torch.cat(preds)
        closed_preds = preds[:, :num_classes, :, :].cpu()
        anomaly_preds = preds[:, num_classes, :, :].cpu()

        pred_mious = mean_iou(torch.argmax(closed_preds, axis=1), labels, num_classes=num_classes)
        mious += pred_mious.tolist()

        if compute_ap:
            pred_aps = binary_average_precision(anomaly_preds, labels == StreetHazardsClasses.ANOMALY)
            aps.append(pred_aps.item())

    return {
        "miou": {
            "mean": np.mean(mious),
            "std": np.std(mious),
        },
        "ap": {
            "mean": np.mean(aps),
            "std": np.std(aps),
        } if compute_ap else None
    }