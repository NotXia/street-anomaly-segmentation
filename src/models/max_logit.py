import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pickle

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.PretrainedSegmenter import PretrainedSegmenter
    


class MaxLogitModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter):
        """
        MaxLogit anomaly detector defined in "Scaling Out-of-Distribution Detection for Real-World Settings" <https://arxiv.org/abs/1911.11132>.
        """
        super().__init__()
        self.segmenter = segmenter.eval()
    
    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs.to(self.segmenter.device))
        ood_scores = -1 * torch.max(logits, dim=1).values
        return logits, ood_scores.unsqueeze(1)
    


class MaxSoftmaxModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter):
        super().__init__()
        self.segmenter = segmenter.eval()
    
    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs.to(self.segmenter.device))
        ood_scores = -1 * torch.max(F.softmax(logits, dim=1), dim=1).values
        return logits, ood_scores.unsqueeze(1)



class StandardizedMaxLogitModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter):
        """
        MaxLogit anomaly detector defined in "Scaling Out-of-Distribution Detection for Real-World Settings" <https://arxiv.org/abs/1911.11132>.
        """
        super().__init__()
        self.segmenter = segmenter.eval()
    
    @torch.no_grad()
    def fit(self, dl_train: torch.utils.data.DataLoader):
        all_maxlogits = []
        all_preds = []

        for images, _ in tqdm(dl_train):
            logits = self.segmenter(images.to(self.segmenter.device))
            segm_preds = torch.argmax(logits, dim=1)

            all_maxlogits.append(logits.cpu().max(dim=1).values)
            all_preds.append(segm_preds.cpu())

        all_maxlogits = torch.concat(all_maxlogits, dim=0)
        all_preds = torch.concat(all_preds, dim=0)

        means = torch.zeros((self.segmenter.num_classes))
        variances = torch.zeros((self.segmenter.num_classes))
        for c in range(self.segmenter.num_classes):
            means[c] = all_maxlogits[all_preds == c].sum() / (all_preds == c).sum()
            variances[c] = ((all_maxlogits[all_preds == c] - means[c].unsqueeze(0))**2).sum() / (all_preds == c).sum()
        
        self.means = means
        self.variances = variances

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs.to(self.segmenter.device))
        segm_preds = torch.argmax(logits, dim=1)
        self.means, self.variances = self.means.to(segm_preds.device), self.variances.to(segm_preds.device)
        
        ood_scores = -1 * (torch.max(logits, dim=1).values - self.means[segm_preds]) / torch.sqrt(self.variances[segm_preds])

        return logits, ood_scores.unsqueeze(1)
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "means": self.means.cpu().tolist(),
                "variances": self.variances.cpu().tolist()
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            fitted_data = pickle.load(f)
            self.means = torch.Tensor(fitted_data["means"])
            self.variances = torch.Tensor(fitted_data["variances"])
