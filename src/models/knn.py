"""
Anomaly detectors based on k-nearest neighbors.

Inspired by the paper "Far Away in the Deep Space: Dense Nearest-Neighbor-Based Out-of-Distribution Detection" <https://arxiv.org/abs/2211.06660v2>.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.PretrainedSegmenter import PretrainedSegmenter
from models.utils.ViTAutoencoder import ViTAutoencoder

from typing import Union



@torch.no_grad()
def extract_hidden_states(
        model: Union[PretrainedSegmenter, ViTAutoencoder], 
        dataloader: torch.utils.data.DataLoader, 
        hidden_states_index: int = -1
    ) -> torch.Tensor:
    """
    Extracts the embedding space of a model on a given dataloader.

    Args:
        model (Union[PretrainedSegmenter, ViTAutoencoder])
        dataloader (torch.utils.data.DataLoader)
        hidden_states_index (int): Index of the layer from which to extract the embeddings if more are available.

    Returns:
        embeddings (torch.Tensor): All the extracted embeddings in the shape num_emb x emb_size.
    """
    all_hidden_states = []
    model.eval()

    for images, _ in tqdm(dataloader):
        if isinstance(model, PretrainedSegmenter):
            _, hidden_states = model(images, return_hidden_states=True)
            embeddings = hidden_states[hidden_states_index]
        elif isinstance(model, ViTAutoencoder):
            _, embeddings = model(images)
        else:
            raise RuntimeError("Unsupported model")
        
        embeddings = embeddings.permute((0, 2, 3, 1)).flatten(start_dim=0, end_dim=2).cpu() # B*H_e*W_e x emb
        all_hidden_states.append(embeddings)

    return torch.concat(all_hidden_states, dim=0)


def _get_ood_scores(knn: NearestNeighbors, embeddings: torch.Tensor):
    """
    Computes the OOD scores for each embedding based on the average distance to the nearest neighbors.

    Args:
        knn (NearestNeighbors): Fittest k-NN model.
        embeddings (torch.Tensor): Embeddings to compute OOD scores for. The expected shape is B x emb x H x W.

    Returns:
        ood_scores (torch.Tensor): Anomaly scores of shape B x H x W.
    """
    embeddings = embeddings.permute((0, 2, 3, 1)) # B x H x W x emb
    batch_size, latent_h, latent_w = embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]
    ood_scores = np.zeros((batch_size, latent_h, latent_w))

    for i in range(len(embeddings)):
        knn_queries = embeddings[i].flatten(start_dim=0, end_dim=1)         # HW x emb
        distances, _ = knn.kneighbors(knn_queries)                          # HW x n_neighbors
        ood_scores[i] = distances.mean(axis=1).reshape(latent_h, latent_w)  # H x W
    
    ood_scores = torch.from_numpy(ood_scores)
    return ood_scores



class KNNModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter, n_neighbors: int=3, hidden_states_index: int=-1):
        """
        Base k-NN anomaly detector that uses the embedding space of a segmenter.
        """
        super().__init__()
        self.segmenter = segmenter.eval()
        self.hidden_states_index = hidden_states_index
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)

    def fit(self, hidden_states: torch.Tensor):
        self.knn.fit(hidden_states)
        return self
    
    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden_states = self.segmenter(inputs, return_hidden_states=True)
        
        ood_scores = _get_ood_scores(self.knn, hidden_states[self.hidden_states_index].cpu())
        ood_scores = F.interpolate(ood_scores.unsqueeze(1), (inputs.shape[2], inputs.shape[3]), mode="bilinear")

        return logits, ood_scores



class KNNModelAE(BaseSegmenterAndDetector, torch.nn.Module):
    """
    K-NN anomaly detector that uses the embedding space of an autoencoder.
    """
    def __init__(self, segmenter: PretrainedSegmenter, ae: ViTAutoencoder, n_neighbors: int=3):
        super().__init__()
        self.segmenter = segmenter.eval()
        self.ae = ae.eval()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)

    def fit(self, hidden_states: torch.Tensor):
        self.knn.fit(hidden_states)
        return self
    
    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs)
        _, embeddings = self.ae(inputs)

        ood_scores = _get_ood_scores(self.knn, embeddings.cpu())
        ood_scores = F.interpolate(ood_scores.unsqueeze(1), (inputs.shape[2], inputs.shape[3]), mode="bilinear")

        return logits, ood_scores
    


class KmeansModel(KNNModel):
    """
    K-NN anomaly detector for segmentation models that uses k-means to reduce the embeddings used by k-NN.
    """
    def __init__(self, segmenter: PretrainedSegmenter, num_clusters, n_neighbors=3, hidden_states_index=-1, seed=42):
        super().__init__(segmenter, n_neighbors, hidden_states_index)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=seed)

    @torch.no_grad()
    def fit(self, hidden_states: torch.Tensor):
        self.kmeans.fit(hidden_states)
        self.knn.fit(self.kmeans.cluster_centers_) # Fit k-NN with centroids.
        return self