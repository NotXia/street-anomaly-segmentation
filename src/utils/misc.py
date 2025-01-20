import lightning as L
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data.dataset import Dataset, Subset


class LitNoValProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True)
    

def dataset_subset(ds: Dataset, num_samples: int, seed=42):
    return Subset(
        ds, 
        np.random.default_rng(seed).choice(range(len(ds)), size=num_samples, replace=False)
    )