import torch



class BaseSegmenterAndDetector:
    def __init__(self):
        super().__init__()

        
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            segmentation (torch.Tensor): (B x C x H x W) tensor with the raw predictions for segmentation.
            anomaly (torch.Tensor): (B x 1 x H x W) tensor with the raw predictions for anomaly.
        """
        raise NotImplementedError()