import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

import torch
import torch.nn.functional as F
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
        more_transforms1 = None,
        more_transforms2 = None,
        random_crop_size: Optional[tuple[int, int]] = None,
        add_random_anomalies: bool = False,
        anomaly_dataset_path: str = "./data_voc"
    ):
        with open(odgt_path, "r") as f:
            odgt_data = json.load(f)

        self.paths = [
            {
                "image": os.path.join(Path(odgt_path).parent, data["fpath_img"]),
                "annotation": os.path.join(Path(odgt_path).parent, data["fpath_segm"]),
            }
            for data in odgt_data 
        ]
        
        self.to_tensor_transform = transforms.ToTensor()
        self.more_transforms1 = more_transforms1 if more_transforms1 is not None else (lambda x: x)
        self.more_transforms2 = more_transforms2 if more_transforms2 is not None else (lambda x: x)

        self.random_crop_size = random_crop_size

        self.add_random_anomalies = add_random_anomalies
        if self.add_random_anomalies:
            try:
                self.ds_anomaly = torchvision.datasets.VOCSegmentation(anomaly_dataset_path, image_set="val")
            except:
                self.ds_anomaly = torchvision.datasets.VOCSegmentation(anomaly_dataset_path, image_set="val", download=True)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]["image"]).convert("RGB")
        annotation = Image.open(self.paths[idx]["annotation"])

        # Apply transforms
        image = self.to_tensor_transform(image)
        image = self.more_transforms1(image)
        annotation = torch.as_tensor(transforms.functional.pil_to_tensor(annotation), dtype=torch.int64) - 1 # Make class indexes start from 0
        annotation = self.more_transforms2(annotation).squeeze(0)

        # Apply same random crop on image and annotation
        if self.random_crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.random_crop_size)
            image = transforms.functional.crop(image, i, j, h, w)
            annotation = transforms.functional.crop(annotation, i, j, h, w)
        
        # Add random anomaly
        if self.add_random_anomalies:
            for i in range(np.random.randint(1, 4)):
                # Determine position of the anomaly
                anomaly_size = (np.random.randint(image.shape[1]*0.1, image.shape[1]*0.3), np.random.randint(image.shape[2]*0.1, image.shape[2]*0.3))
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=anomaly_size)

                # Choose random image from Pascal VOC
                possible_classes = []
                while len(possible_classes) == 0: # In some cases there are no classes available
                    anomaly_idx = np.random.randint(0, len(self.ds_anomaly))
                    anomaly_image = self.to_tensor_transform(self.ds_anomaly[anomaly_idx][0])
                    anomaly_annot = torch.from_numpy(np.array(self.ds_anomaly[anomaly_idx][1])).unsqueeze(0)
                    possible_classes = np.unique(anomaly_annot)[1:-1] # Ignore 0 and 255

                # Select random class in chosen image and resize to the resolution of the crop
                anomaly_class = np.random.choice(possible_classes)
                anomaly_image = F.interpolate(anomaly_image.unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
                anomaly_annot = F.interpolate(anomaly_annot.unsqueeze(0), size=(h, w), mode="nearest").squeeze((0, 1))

                # Insert anomaly
                image[:, i:i+h, j:j+w][:, anomaly_annot == anomaly_class] = anomaly_image[:, anomaly_annot == anomaly_class]
                annotation[i:i+h, j:j+w][anomaly_annot == anomaly_class] = StreetHazardsClasses.ANOMALY

        return image, annotation

    def __len__(self):
        return len(self.paths)
    

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

def visualize_annotation(annotation_img: np.ndarray|torch.Tensor, ax=None):
    """
    Adapted from https://github.com/CVLAB-Unibo/ml4cv-assignment/blob/master/utils/visualize.py
    """
    if ax is None: ax = plt.gca()
    annotation_img = np.asarray(annotation_img)
    img_new = np.zeros((*annotation_img.shape, 3))

    for index, color in enumerate(COLORS):
        img_new[annotation_img == index] = color

    ax.imshow(img_new / 255.0)
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_scene(image: np.ndarray|torch.Tensor, ax=None):
    if ax is None: ax = plt.gca()
    image = np.asarray(image)
    ax.imshow(np.moveaxis(image, 0, -1))
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_anomaly(anomaly_map: np.ndarray|torch.Tensor, alpha=1, ax=None, colorbar=False):
    if ax is None: ax = plt.gca()
    im = ax.imshow(anomaly_map, alpha=alpha, cmap="turbo")
    ax.set_xticks([])
    ax.set_yticks([])

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)        
        plt.gcf().colorbar(im, cax=cax, orientation="vertical")

def show_results(image, labels, segm_preds, anomaly_map, figsize=(18, 2), title=""):
    plt.figure(figsize=figsize)
    plt.suptitle(title)
    plt.subplot(1, 5, 1)
    visualize_scene(image)
    plt.subplot(1, 5, 2)
    visualize_annotation(labels)
    plt.subplot(1, 5, 3)
    visualize_annotation(segm_preds)
    plt.subplot(1, 5, 4)
    visualize_scene(image)
    visualize_anomaly(anomaly_map, alpha=0.6)
    plt.subplot(1, 5, 5)
    visualize_anomaly(anomaly_map, colorbar=True)
    plt.show()