import torch
import torch.nn.functional as F
import lightning as L

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.PretrainedSegmenter import PretrainedSegmenter
from utils.street_hazards import StreetHazardsClasses
from utils.eval import AccumulatorMIoU



class MaxLogitModel(BaseSegmenterAndDetector, L.LightningModule):
    def __init__(self, segmenter: PretrainedSegmenter, optimizer_args={}, num_classes=len(StreetHazardsClasses)-1):
        super().__init__()
        self.segmenter = segmenter
        self.optimizer_args = optimizer_args
        self.num_classes = num_classes
        self.train_miou_acc = AccumulatorMIoU(num_classes, None)
        self.val_miou_acc = AccumulatorMIoU(num_classes, None)
        
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.segmenter.parameters(), **self.optimizer_args)
    
    
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmenter(inputs)
        anom_pred = -1 * torch.max(logits, dim=1).values
        return logits, anom_pred


    def __train_forward(self, images, labels):
        logits = self.segmenter(images)
        loss = F.cross_entropy(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self.__train_forward(images, labels)

        self.train_miou_acc.add(torch.argmax(logits, dim=1), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self.__train_forward(images, labels)

        self.val_miou_acc.add(torch.argmax(logits, dim=1), labels)
        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        self.log("train_miou", self.train_miou_acc.compute())
        self.log("val_miou", self.val_miou_acc.compute())
        metrics = self.trainer.callback_metrics

        print(
            f"Epoch {self.current_epoch} --- train_loss: {metrics['train_loss'].item():.4f} -- val_loss: {metrics['val_loss'].item():.4f}"
            f" | train_miou: {metrics['train_miou'].item():.4f} -- val_miou: {metrics['val_miou'].item():.4f}"
        )

        self.train_miou_acc.reset()
        self.val_miou_acc.reset()