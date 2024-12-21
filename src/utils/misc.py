import numpy as np
import torch



def sliding_window_inference(model: torch.nn.Module, image: torch.Tensor, patch_size: tuple[int, int], device) -> torch.Tensor:
    """
    Inspired from: https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6
    """
    # Divide image into patches
    patches = image.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1])
    patches_h = patches.shape[1]
    patches_w = patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, 144, 256) # Move patches shape in front (i.e., form a batch)

    pred_patches = model(patches.to(device))

    # Reconstruct original spatial dimension
    out_channels = pred_patches.shape[1]
    pred_patches = pred_patches.reshape(patches_h, patches_w, out_channels, patch_size[0], patch_size[1]).permute(2, 0, 3, 1, 4)
    preds = pred_patches.reshape((out_channels, image.shape[1], image.shape[2]))

    return preds
