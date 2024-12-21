import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.segmentation import mean_iou



class SigmoidModel(L.LightningModule):
    def __init__(self, segmentation_model, optimizer_args={}, num_classes=13):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.optimizer_args = optimizer_args
        self.num_classes = num_classes

    def forward(self, inputs):
        segm_preds = self.segmentation_model(inputs)
        segm_preds = F.sigmoid(segm_preds)
        anom_preds = 1 - torch.max(segm_preds, dim=1).values
        return torch.cat([segm_preds, anom_preds.unsqueeze(1)], dim=1)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.segmentation_model.parameters(), **self.optimizer_args)


    def _loss(self, segm_preds, anom_preds, labels):
        loss = 0
        for i in range(segm_preds.shape[1]):
            loss += F.binary_cross_entropy(segm_preds[:, i], (labels == i).type(torch.float32))
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        segm_preds, anom_preds = preds[:, :self.num_classes, :, :], preds[:, self.num_classes, :, :]
        loss = self._loss(segm_preds, anom_preds, labels)

        self.log("train_miou", mean_iou(torch.argmax(segm_preds, axis=1), labels, num_classes=self.num_classes).mean())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        segm_preds, anom_preds = preds[:, :self.num_classes, :, :], preds[:, self.num_classes, :, :]
        loss = self._loss(segm_preds, anom_preds, labels)

        self.log("val_miou", mean_iou(torch.argmax(segm_preds, axis=1), labels, num_classes=self.num_classes).mean())
        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(
            f"Epoch {self.current_epoch} --- train_loss: {metrics['train_loss'].item():.5f} | train_miou: {metrics['train_miou'].item():.5f}"
            f" | val_loss: {metrics['val_loss'].item():.5f} | val_miou: {metrics['val_miou'].item():.5f}"
        )



class ExpSigmoidModel(SigmoidModel):
    def forward(self, inputs):
        segm_preds = self.segmentation_model(inputs)
        segm_preds = F.sigmoid(torch.sign(segm_preds) * torch.exp(segm_preds))
        anom_preds = 1 - torch.max(segm_preds, dim=1).values
        return torch.cat([segm_preds, anom_preds.unsqueeze(1)], dim=1)



class MaskedSigmoidModel(SigmoidModel):
    def _loss(self, segm_preds, anom_preds, labels):
        loss = 0
        for i in range(segm_preds.shape[1]):
            masked_segm_preds = segm_preds[:, i] * torch.logical_or(torch.logical_and(segm_preds[:, i] <= anom_preds, (labels == i)), (labels != i))
            loss += F.binary_cross_entropy(masked_segm_preds, (labels == i).type(torch.float32))
        return loss
