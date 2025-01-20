import lightning as L
from tqdm.auto import tqdm


class LitNoValProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True)