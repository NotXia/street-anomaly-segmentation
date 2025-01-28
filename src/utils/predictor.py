"""
Wrapper classes used for inference that output segmentation maps and anomaly scores.
"""


from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
import numpy as np
try:
    from fast_slic.avx2 import SlicAvx2 as Slic
except:
    from fast_slic import Slic
import torch


class BasePredictor:
    def __init__(self, model: BaseSegmenterAndDetector):
        super().__init__()
        self.model = model.eval()

    def to(self, device):
        self.model.to(device)

    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            segmentation_map (torch.Tensor): Shape H x W
            anomaly_predicitions (torch.Tensor): Shape H x W
        """
        raise NotImplementedError()

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        segm_preds, ood_scores = self.forward(inputs)
        return segm_preds, ood_scores
    

class StandardPredictor(BasePredictor):
    def __init__(self, model: BaseSegmenterAndDetector):
        """
        Predictor that applies argmax on segmentation logits and leaves OOD scores as-is.
        """
        super().__init__(model)

    @torch.no_grad()
    def forward(self, inputs):
        segm_logits, ood_scores = self.model(inputs)
        return torch.argmax(segm_logits, dim=1), ood_scores.squeeze(1)


def _prepare_image_for_slic(image):
    image = np.moveaxis(image.numpy(), 0, -1) # H x W x C
    image = (image*255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    return image

class SuperpixelPredictor(BasePredictor):
    def __init__(self, 
            model: BaseSegmenterAndDetector, 
            apply_on_segmentation = False, 
            apply_on_anomaly = True, 
            slic_num_components = 200, 
            slic_compactness = 20,
        ):
        super().__init__(model)
        assert apply_on_segmentation or apply_on_anomaly, "Nothing to segment"
        self.apply_on_segmentation = apply_on_segmentation
        self.apply_on_anomaly = apply_on_anomaly
        self.slic_num_components = slic_num_components
        self.slic_compactness = slic_compactness
    

    @torch.no_grad()
    def forward(self, inputs):
        slic = Slic(num_components=self.slic_num_components, compactness=self.slic_compactness) # Must be recreated every time for determinism
        segm_logits, ood_scores = self.model(inputs)
        segm_maps = torch.argmax(segm_logits, dim=1)
        ood_scores = ood_scores.squeeze(1)

        for i in range(len(inputs)):
            segments = slic.iterate(_prepare_image_for_slic(inputs[i].cpu()))

            for j in np.unique(segments):
                if self.apply_on_segmentation:
                    # The segmentation class of a superpixel is the most frequent class within it
                    segm_maps[i, segments == j] = segm_maps[i, segments == j].mode().values
                if self.apply_on_anomaly:
                    # The anomaly score of a superpixel is the average of the scores within it
                    ood_scores[i, segments == j] = ood_scores[i, segments == j].mean()

        return segm_maps, ood_scores