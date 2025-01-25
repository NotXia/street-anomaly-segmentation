from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
import numpy as np
try:
    from fast_slic.avx2 import SlicAvx2 as Slic
except:
    from fast_slic import Slic
import torch
import torch.nn.functional as F


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
        super().__init__(model)

    @torch.no_grad()
    def forward(self, inputs):
        segm_logits, ood_scores = self.model(inputs)
        return torch.argmax(segm_logits, dim=1), ood_scores.squeeze(1)


def _prepare_image_for_slic(image):
    image = np.moveaxis(image.numpy(), 0, -1)
    image = (image*255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    return image

class SmoothedPredictor(BasePredictor):
    def __init__(self, 
            model: BaseSegmenterAndDetector, 
            smooth_segmentation = False, 
            smooth_anomaly = True, 
            slic_num_components = 200, 
            slic_compactness = 20,
        ):
        super().__init__(model)
        assert smooth_segmentation or smooth_anomaly, "Nothing to smooth"
        self.smooth_segmentation = smooth_segmentation
        self.smooth_anomaly = smooth_anomaly
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
                if self.smooth_segmentation:
                    segm_maps[i, segments == j] = segm_maps[i, segments == j].mode().values
                if self.smooth_anomaly:
                    ood_scores[i, segments == j] = ood_scores[i, segments == j].mean()

        return segm_maps, ood_scores