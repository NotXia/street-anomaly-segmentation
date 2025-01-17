import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor, ViTConfig

from typing import Union



def _upsample_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class ViTAutoencoder(nn.Module):
    def __init__(self, vit_model: Union[str, ViTConfig], decoder_start_size=128, out_channels=3):
        super().__init__()
        if isinstance(vit_model, str):
            self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)
            self.vit = ViTModel.from_pretrained(vit_model, add_pooling_layer=False)
        elif isinstance(vit_model, ViTConfig):
            self.vit_processor = ViTImageProcessor(vit_model)
            self.vit = ViTModel(vit_model, add_pooling_layer=False)
        else:
            raise RuntimeError("Unknown model")

        self.decoder = nn.Sequential(
            _upsample_block(self.vit.config.hidden_size, decoder_start_size, kernel_size=3, stride=1, padding=1),
            _upsample_block(decoder_start_size, decoder_start_size, kernel_size=5, stride=2, padding=1),

            _upsample_block(decoder_start_size, decoder_start_size//2, kernel_size=3, stride=1, padding=1),
            _upsample_block(decoder_start_size//2, decoder_start_size//2, kernel_size=3, stride=2, padding=1),

            _upsample_block(decoder_start_size//2, decoder_start_size//4, kernel_size=3, stride=1, padding=1),
            _upsample_block(decoder_start_size//4, decoder_start_size//4, kernel_size=3, stride=2, padding=1),

            _upsample_block(decoder_start_size//4, decoder_start_size//4, kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=decoder_start_size//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, images):
        orig_shape = (images.shape[2], images.shape[3])
        images = F.interpolate(images, size=(self.vit.config.image_size, self.vit.config.image_size), mode="bilinear")
        
        vit_inputs = self.vit_processor(images=images, return_tensors="pt", do_resize=False, do_rescale=False).to("cuda")
        vit_outputs = self.vit(**vit_inputs)
        latents = vit_outputs["last_hidden_state"][:, 1:, :] # (B x patches x hidden_size) without CLS

        # Make latent image-like (B x hidden_size x H_patches x W_patches)
        latents_by_patch = torch.permute(latents, (0, 2, 1))
        latents_by_patch = latents_by_patch.reshape((-1, latents_by_patch.shape[1], self.vit.config.image_size // self.vit.config.patch_size, self.vit.config.image_size // self.vit.config.patch_size))

        reconstruction = self.decoder(latents_by_patch)
        reconstruction = F.interpolate(reconstruction, size=orig_shape, mode="bilinear")
        return reconstruction, latents_by_patch
    

    def get_hidden_size(self):
        return self.vit.config.hidden_size