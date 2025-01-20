from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
import numpy as np
from fast_slic import Slic
import torch



class BasePredictor:
    def __init__(self, model: BaseSegmenterAndDetector):
        super().__init__()
        self.model = model.eval()

    def to(self, device):
        self.model.to(device)

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            segmentation_map (torch.Tensor)
            anomaly_predicitions (torch.Tensor)
        """
        raise NotImplementedError()
    

class StandardPredictor(BasePredictor):
    def __init__(self, model: BaseSegmenterAndDetector):
        super().__init__(model)

    def __call__(self, inputs):
        with torch.no_grad():
            segm_logits, anom_preds = self.model(inputs)
        return torch.argmax(segm_logits, dim=1), anom_preds.squeeze(1)


class SmoothedPredictor(BasePredictor):
    def __init__(self, model: BaseSegmenterAndDetector, smooth_segmentation=True, smooth_anomaly=True):
        super().__init__(model)
        assert smooth_segmentation or smooth_anomaly, "Nothing to smooth"
        self.slic = Slic(num_components=500, compactness=1)
        self.smooth_segmentation = smooth_segmentation
        self.smooth_anomaly = smooth_anomaly

    def __call__(self, inputs):
        with torch.no_grad():
            segm_logits, anom_preds = self.model(inputs)
        segm_maps = torch.argmax(segm_logits, dim=1)
        anom_preds = anom_preds.squeeze(1)

        for i in range(len(inputs)):
            image = inputs[i].cpu()
            image = np.moveaxis(image.numpy(), 0, -1)
            image = (image*255).astype(np.uint8)
            image = np.ascontiguousarray(image)
            segments = self.slic.iterate(image)

            for j in np.unique(segments):
                if self.smooth_segmentation:
                    segm_maps[i][segments == j] = segm_maps[i][segments == j].mode().values
                if self.smooth_anomaly:
                    anom_preds[i][segments == j] = anom_preds[i][segments == j].mean()

        return segm_maps, anom_preds