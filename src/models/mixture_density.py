import torch
import torch.nn.functional as F
import lightning as L
import numpy as np

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.ViTAutoencoder import ViTAutoencoder
from models.utils.PretrainedSegmenter import PretrainedSegmenter
from models.utils.GaussianMixtureDensityNetwork import GaussianMixtureDensityNetwork
from utils.street_hazards import StreetHazardsClasses
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from utils.eval import AccumulatorMIoU



def _reshape_latent_for_mdn(latents):
    """
    Reshapes a latent from (B x hidden_size x H x W) to (B x flat_spatial_dimension x hidden_size)
    """
    latents = torch.permute(latents, (0, 2, 3, 1))
    latents = torch.flatten(latents, start_dim=1, end_dim=2)
    return latents

def _reshape_probs_as_image(probs):
    """
    Reshapes a probability map from (B x flat_spatial_dimension) to (B x 1 x H x W).
    The image is assumed to be a square.
    """
    probs = probs.reshape(probs.shape[0], 1, int(np.sqrt(probs.shape[1])), int(np.sqrt(probs.shape[1])))
    return probs



class AutoencoderMDN(BaseSegmenterAndDetector, L.LightningModule):
    def __init__(self, autoencoder: ViTAutoencoder, num_gaussians: int, pretrained_segmenter: PretrainedSegmenter, optimizer_args={}):
        super().__init__()
        self.autoencoder = autoencoder
        self.mdn = GaussianMixtureDensityNetwork(self.autoencoder.get_hidden_sizes()[-1], num_gaussians)
        self.segmenter = pretrained_segmenter.eval() # The segmenter is not trained in this module
        self.optimizer_args = optimizer_args
    

    def configure_optimizers(self):
        return torch.optim.AdamW([*self.autoencoder.parameters(), *self.mdn.parameters()], **self.optimizer_args)

    def autoencoder_requires_grad(self, requires_grad):
        self.autoencoder.requires_grad_(requires_grad)
        self.autoencoder.train() if requires_grad else self.autoencoder.eval()
        self.train_ae = requires_grad

    def mdn_requires_grad(self, requires_grad):
        self.mdn.requires_grad_(requires_grad)
        self.mdn.train() if requires_grad else self.mdn.eval()
        self.train_mdn = requires_grad
    

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        segm_logits = self.segmenter(inputs)
        _, latents = self(inputs)
        latents = _reshape_latent_for_mdn(latents)
        weights, means, logvars = self.mdn(latents)
        anom_log_probs = GaussianMixtureDensityNetwork.loss(latents, weights, means, logvars, reduction="none")
        anom_log_probs = _reshape_probs_as_image(anom_log_probs)
        anom_log_probs = F.interpolate(anom_log_probs, size=(inputs.shape[2], inputs.shape[3]), mode="bilinear")
        return segm_logits, anom_log_probs


    def __train_forward(self, images):
        reconstructions, latents = self.autoencoder(images)
        latents = _reshape_latent_for_mdn(latents)
        weights, means, logvars = self.mdn(latents) if self.train_mdn else (0, 0, 0)
        return reconstructions, latents, weights, means, logvars

    def __loss(self, orig_images, reconstructions, latents, weights, means, logvars):
        loss_mse = 5 * F.mse_loss(reconstructions, orig_images)
        loss_ssim = ( 1-ssim(reconstructions, orig_images, data_range=(0.0, 1.0)) )
        loss_likelihood = GaussianMixtureDensityNetwork.loss(latents, weights, means, logvars, reduction="mean") if self.train_mdn else 0
        loss = 0
        if self.train_ae: loss += loss_mse + loss_ssim
        if self.train_mdn: loss += loss_likelihood
        return loss, loss_mse, loss_ssim, loss_likelihood

    def training_step(self, batch, batch_idx):
        images, _ = batch
        reconstructions, latents, weights, means, logvars = self.__train_forward(images)
        loss, loss_mse, loss_ssim, loss_likelihood = self.__loss(images, reconstructions, latents, weights, means, logvars)

        self.log("train_mse", loss_mse)
        self.log("train_ssim", loss_ssim)
        self.log("train_ll", loss_likelihood)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstructions, latents, weights, means, logvars = self.__train_forward(images)
        loss, loss_mse, loss_ssim, loss_likelihood = self.__loss(images, reconstructions, latents, weights, means, logvars)

        self.log("val_mse", loss_mse)
        self.log("val_ssim", loss_ssim)
        self.log("val_ll", loss_likelihood)
        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(
            f"Epoch {self.current_epoch} --- train_loss: {metrics['train_loss'].item():.4f} -- val_loss: {metrics['val_loss'].item():.4f}"
            f" | train_mse: {metrics['train_mse'].item():.4f} -- val_mse: {metrics['val_mse'].item():.4f}"
            f" | train_ssim: {metrics['train_ssim'].item():.4f} -- val_ssim: {metrics['val_ssim'].item():.4f}"
            f" | train_ll: {metrics['train_ll'].item():.4f} -- val_ll: {metrics['val_ll'].item():.4f}"
        )



class SegmenterMDN(BaseSegmenterAndDetector, L.LightningModule):
    def __init__(self, segmenter: PretrainedSegmenter, num_gaussians: int, optimizer_args={}, num_classes=len(StreetHazardsClasses)-1):
        super().__init__()
        self.segmenter = segmenter
        self.mdn = GaussianMixtureDensityNetwork(self.segmenter.get_hidden_sizes()[-1], num_gaussians)
        self.optimizer_args = optimizer_args
        self.train_miou_acc = AccumulatorMIoU(num_classes, None)
        self.val_miou_acc = AccumulatorMIoU(num_classes, None)


    def configure_optimizers(self):
        return torch.optim.AdamW([*self.segmenter.parameters(), *self.mdn.parameters()], **self.optimizer_args)

    def segmenter_requires_grad(self, requires_grad):
        self.segmenter.requires_grad_(requires_grad)
        self.segmenter.train() if requires_grad else self.segmenter.eval()
        self.train_segm = requires_grad

    def mdn_requires_grad(self, requires_grad):
        self.mdn.requires_grad_(requires_grad)
        self.mdn.train() if requires_grad else self.mdn.eval()
        self.train_mdn = requires_grad
    

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        segm_logits, hidden_states = self.segmenter(inputs, return_hidden_states=True)
        latents = _reshape_latent_for_mdn(hidden_states[-1])
        weights, means, logvars = self.mdn(latents)
        anom_log_probs = GaussianMixtureDensityNetwork.loss(latents, weights, means, logvars, reduction="none")
        anom_log_probs = _reshape_probs_as_image(anom_log_probs)
        anom_log_probs = F.interpolate(anom_log_probs, size=(inputs.shape[2], inputs.shape[3]), mode="bilinear")
        return segm_logits, anom_log_probs


    def __train_forward(self, images):
        semg_logits, hidden_states = self.segmenter(images, return_hidden_states=True)
        latents = _reshape_latent_for_mdn(hidden_states[-1])
        weights, means, logvars = self.mdn(latents) if self.train_mdn else (0, 0, 0)
        return semg_logits, latents, weights, means, logvars

    def __loss(self, labels, segm_logits, latents, weights, means, logvars):
        loss_ce = F.cross_entropy(segm_logits, labels)
        loss_likelihood = GaussianMixtureDensityNetwork.loss(latents, weights, means, logvars, reduction="mean") if self.train_mdn else 0
        loss = 0
        if self.train_segm: loss += loss_ce
        if self.train_mdn: loss += loss_likelihood
        return loss, loss_ce, loss_likelihood

    def training_step(self, batch, batch_idx):
        images, labels = batch
        semg_logits, latents, weights, means, logvars = self.__train_forward(images)
        loss, loss_ce, loss_likelihood = self.__loss(labels, semg_logits, latents, weights, means, logvars)
        
        self.train_miou_acc.add(torch.argmax(semg_logits, dim=1), labels)
        self.log("train_ce", loss_ce)
        self.log("train_ll", loss_likelihood)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        semg_logits, latents, weights, means, logvars = self.__train_forward(images)
        loss, loss_ce, loss_likelihood = self.__loss(labels, semg_logits, latents, weights, means, logvars)
        
        self.val_miou_acc.add(torch.argmax(semg_logits, dim=1), labels)
        self.log("val_ce", loss_ce)
        self.log("val_ll", loss_likelihood)
        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        self.log("train_miou", self.train_miou_acc.compute())
        self.log("val_miou", self.val_miou_acc.compute())
        metrics = self.trainer.callback_metrics

        print(
            f"Epoch {self.current_epoch} --- train_loss: {metrics['train_loss'].item():.4f} -- val_loss: {metrics['val_loss'].item():.4f}"
            f" | train_ce: {metrics['train_ce'].item():.4f} -- val_ce: {metrics['val_ce'].item():.4f}"
            f" | train_miou: {metrics['train_miou'].item():.4f} -- val_miou: {metrics['val_miou'].item():.4f}"
            f" | train_ll: {metrics['train_ll'].item():.4f} -- val_ll: {metrics['val_ll'].item():.4f}"
        )

        self.train_miou_acc.reset()
        self.val_miou_acc.reset()