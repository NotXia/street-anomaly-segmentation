import torch
from torchvision.models.segmentation import \
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights, \
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from typing import Literal



class PretrainedSegmentationModel(torch.nn.Module):
    def __init__(
            self,
            model: Literal["deeplabv3_mobilenet", "deeplabv3_mobilenet_coco", "deeplabv3_resnet50"],
            num_classes: int = 13
        ):
        super().__init__()
        match model:
            case "deeplabv3_mobilenet":
                self.segmentation_model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
            case "deeplabv3_mobilenet_coco":
                self.segmentation_model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
                old_conv = self.segmentation_model.classifier[4]
                self.segmentation_model.classifier[4] = torch.nn.Conv2d(old_conv.in_channels, num_classes, old_conv.kernel_size, old_conv.stride)
            case "deeplabv3_resnet50":
                self.segmentation_model = deeplabv3_resnet50(num_classes=num_classes)
            case _:
                raise RuntimeError("Unknown pre-trained model")

    def forward(self, input):
        preds = self.segmentation_model(input)
        return preds["out"]