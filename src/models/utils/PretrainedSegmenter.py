import torch
import torch.nn.functional as F
import lightning as L
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
import transformers
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from utils.street_hazards import StreetHazardsClasses
from utils.eval import AccumulatorMIoU
from typing import Optional, Union, Literal



class PretrainedSegmenter(L.LightningModule):
    def __init__(self, 
        optimizer_args: dict, 
        image_size: Optional[tuple[int, int]], 
        num_classes: int, 
        class_weights: Optional[torch.Tensor],
        multi_label: bool
    ):
        super().__init__()
        self.model = None
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)
        self.multi_label = multi_label
        self.optimizer_args = optimizer_args
        self.train_miou_acc = AccumulatorMIoU(num_classes, None)
        self.val_miou_acc = AccumulatorMIoU(num_classes, None)


    def _forward(self, images) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            logits
            hidden_states (if any)
        """
        raise NotImplementedError()


    def forward(self, images, return_hidden_states=False):
        if self.image_size:
            original_shape = (images.shape[2], images.shape[3])
            images = F.interpolate(images, self.image_size, mode="bilinear")

        logits, hidden_states = self._forward(images)

        if self.image_size:
            logits = F.interpolate(logits, original_shape, mode="bilinear")

        return (logits, hidden_states) if return_hidden_states else logits


    def get_hidden_sizes(self):
        raise NotImplementedError()
    

    @staticmethod
    def get(
        model_name: Union[Literal["deeplabv3_mobilenet", "deeplabv3_resnet50"], str], 
        optimizer_args: dict = {}, 
        image_size: Optional[tuple[int, int]] = None, 
        num_classes: int = len(StreetHazardsClasses)-1,
        class_weights: Optional[torch.Tensor] = None,
        multi_label: bool = False
    ):
        match model_name:
            case "deeplabv3_mobilenet":
                return _PretrainedPytorchSegmenter(model_name, optimizer_args, image_size, num_classes, class_weights, multi_label)
            case "deeplabv3_resnet50":
                return _PretrainedPytorchSegmenter(model_name, optimizer_args, image_size, num_classes, class_weights, multi_label)
            case _:
                return _PretrainedHuggingFaceSegmenter(model_name, optimizer_args, image_size, num_classes, class_weights, multi_label)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        return optimizer

    def __loss(self, logits, labels):
        if self.multi_label:
            loss = 0
            for c in range(self.num_classes):
                loss += F.binary_cross_entropy_with_logits(logits[:, c], (labels == c).type(torch.float32))
            return loss
        else:
            return F.cross_entropy(logits, labels, weight=self.class_weights)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.__loss(logits, labels)

        self.train_miou_acc.add(torch.argmax(logits, dim=1), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.__loss(logits, labels)

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



class _PretrainedPytorchSegmenter(PretrainedSegmenter):
    def __init__(self, model_name, optimizer_args, image_size, num_classes, class_weights, multi_label):
        super().__init__(optimizer_args, image_size, num_classes, class_weights, multi_label)
        self.model_name = model_name
        match model_name:
            case "deeplabv3_mobilenet":
                self.model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
            case "deeplabv3_resnet50":
                self.model = deeplabv3_resnet50(num_classes=num_classes)
            case _:
                raise NotImplementedError("Model not available")

    def _forward(self, images):
        embeddings = self.model.backbone(images)["out"]
        logits = self.model.classifier(embeddings)
        return logits, [embeddings]
        
    def get_hidden_sizes(self):
        match self.model_name:
            case "deeplabv3_mobilenet": return 960
            case "deeplabv3_resnet50": return 2048


class _PretrainedHuggingFaceSegmenter(PretrainedSegmenter):
    def __init__(self, model_name, optimizer_args, image_size, num_classes, class_weights, multi_label):
        super().__init__(optimizer_args, image_size, num_classes, class_weights, multi_label)
        transformers.utils.logging.set_verbosity_error() # Suppress warning for non-trained head
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.image_size = (self.model.config.image_size, self.model.config.image_size)
        transformers.utils.logging.set_verbosity_warning()

    def _forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=False).to(self.model.device)
        outputs = self.model(
            **inputs,
            output_hidden_states = True
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        return logits, hidden_states
        
    def get_hidden_sizes(self):
        return self.model.config.neck_hidden_sizes[:-1]
