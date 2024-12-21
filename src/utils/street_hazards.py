import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from enum import IntEnum
from typing import Optional


class StreetHazardsClasses(IntEnum):
    UNLABELED       = 0
    BUILDING        = 1
    FENCE           = 2
    OTHER           = 3
    PEDESTRIAN      = 4
    POLE            = 5
    ROAD_LINE       = 6
    ROAD            = 7
    SIDEWALK        = 8
    VEGETATION      = 9
    CAR             = 10
    WALL            = 11
    TRAFFIC_SIGN    = 12
    ANOMALY         = 13


class StreetHazardsDataset(Dataset):
    def __init__(
        self,
        odgt_path: str,
        more_transforms = None,
        patch_size: tuple[int, int] = (720, 1280),
        image_size: tuple[int, int] = (720, 1280)
    ):
        assert (image_size[0] % patch_size[0] == 0) and (image_size[1] % patch_size[1] == 0)
        with open(odgt_path, "r") as f:
            odgt_data = json.load(f)

        self.paths = [
            {
                "image": os.path.join(Path(odgt_path).parent, data["fpath_img"]),
                "annotation": os.path.join(Path(odgt_path).parent, data["fpath_segm"]),
            }
            for data in odgt_data 
        ]
        
        self.image_transforms = transforms.Compose([ transforms.ToTensor() ])
        self.more_transforms = more_transforms if more_transforms is not None else (lambda x: x)
        
        self.patch_size = patch_size
        self.patches_per_row = image_size[1] // patch_size[1]
        self.patches_per_col = image_size[0] // patch_size[0]
        self.patches_per_image = self.patches_per_row * self.patches_per_col

    def __getitem__(self, idx):
        path_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        image = Image.open(self.paths[path_idx]["image"]).convert("RGB")
        annotation = Image.open(self.paths[path_idx]["annotation"])

        # Apply transforms
        image = self.image_transforms(image)
        image = self.more_transforms(image)
        annotation = torch.as_tensor(transforms.functional.pil_to_tensor(annotation), dtype=torch.int64) - 1 # Make class indexes start from 0
        annotation = self.more_transforms(annotation).squeeze(0)

        # Determine patch
        offset_h = (patch_idx // self.patches_per_row) * self.patch_size[0]
        offset_w = (patch_idx % self.patches_per_col) * self.patch_size[1]
        image = image[:, offset_h:offset_h+self.patch_size[0], offset_w:offset_w+self.patch_size[1]]
        annotation = annotation[offset_h:offset_h+self.patch_size[0], offset_w:offset_w+self.patch_size[1]]

        return image, annotation

    def __len__(self):
        return len(self.paths) * self.patches_per_image


COLORS = np.array([
    [  0,   0,   0],  # unlabeled    =   0,
    [ 70,  70,  70],  # building     =   1,
    [190, 153, 153],  # fence        =   2, 
    [250, 170, 160],  # other        =   3,
    [220,  20,  60],  # pedestrian   =   4, 
    [153, 153, 153],  # pole         =   5,
    [157, 234,  50],  # road line    =   6, 
    [128,  64, 128],  # road         =   7,
    [244,  35, 232],  # sidewalk     =   8,
    [107, 142,  35],  # vegetation   =   9, 
    [  0,   0, 142],  # car          =  10,
    [102, 102, 156],  # wall         =  11, 
    [220, 220,   0],  # traffic sign =  12,
    [ 60, 250, 240],  # anomaly      =  13,
])

def visualize_annotation(annotation_img: np.ndarray|torch.Tensor):
    """
    Adapted from https://github.com/CVLAB-Unibo/ml4cv-assignment/blob/master/utils/visualize.py
    """
    annotation_img = np.asarray(annotation_img)
    img_new = np.zeros((*annotation_img.shape, 3))

    for index, color in enumerate(COLORS):
        img_new[annotation_img == index] = color

    plt.imshow(img_new / 255.0)
    plt.xticks([])
    plt.yticks([])

def visualize_scene(img: np.ndarray|torch.Tensor):
    img = np.asarray(img)
    plt.imshow(np.moveaxis(img, 0, -1))
    plt.xticks([])
    plt.yticks([])