import torch

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.PretrainedSegmenter import PretrainedSegmenter



class MaxLogitModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter):
        """
        Max-logit anomaly detector defined in "Scaling Out-of-Distribution Detection for Real-World Settings" <https://arxiv.org/abs/1911.11132>.
        """
        super().__init__()
        self.segmenter = segmenter.eval()
    
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs)
        anom_pred = -1 * torch.max(logits, dim=1).values
        return logits, anom_pred.unsqueeze(1)