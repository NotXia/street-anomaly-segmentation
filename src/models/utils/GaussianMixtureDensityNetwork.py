import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from typing import Literal



class GaussianMixtureDensityNetwork(torch.nn.Module):
    def __init__(self, emb_size: int, num_gaussians: int):
        """
        Gaussian mixture model for image embeddings.

        Args:
            emb_size (int): Dimensionality of the input embeddings.
            num_gaussians (int): Number of Gaussian components
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_gaussians = num_gaussians
        self.proj_weights = nn.Linear(emb_size, num_gaussians, bias=False)
        self.proj_means = nn.Linear(emb_size, num_gaussians*emb_size, bias=False)
        self.proj_variances = nn.Linear(emb_size, num_gaussians*emb_size, bias=False)


    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): (B x P x emb_size) where P is the flattened spatial dimension of the image.
        """
        batch_size = inputs.shape[0]
        num_patches = inputs.shape[1]

        weights = F.softmax(self.proj_weights(inputs), dim=-1)
        means = self.proj_means(inputs).view(batch_size, num_patches, self.emb_size, self.num_gaussians)
        logvars = F.softplus(self.proj_variances(inputs)).view(batch_size, num_patches, self.emb_size, self.num_gaussians)
        
        return weights, means, logvars
        

    @staticmethod
    def loss(x, weights, means, logvars, reduction: Literal["none", "mean"]="none"):
        x = x.unsqueeze(-1).expand_as(logvars)
        log_probs = -1 * Normal(means, torch.exp(logvars)).log_prob(x).mean(2) # Aggregate emb_size. Mean is better to uniform varying sizes.
        log_gmm = (weights * log_probs).sum(2)

        match reduction:
            case "none":
                return log_gmm
            case "mean":
                return torch.mean( torch.mean(log_gmm, 1) )