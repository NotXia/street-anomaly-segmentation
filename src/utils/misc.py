import lightning as L
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data.dataset import Dataset, Subset


class LitNoValProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        """
        Suppresses the tqdm bar for validation.
        """
        return tqdm(disable=True)
    

def dataset_subset(dataset: Dataset, num_samples: int, seed=42):
    """
    Returns a subset of a dataset sampled uniformly.
    """
    return Subset(
        dataset, 
        np.random.default_rng(seed).choice(range(len(dataset)), size=num_samples, replace=False)
    )