import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.utils.BaseSegmenterAndDetector import BaseSegmenterAndDetector
from models.utils.PretrainedSegmenter import PretrainedSegmenter



class SHEModel(BaseSegmenterAndDetector, torch.nn.Module):
    def __init__(self, segmenter: PretrainedSegmenter, hidden_states_index: int=-1):
        """
        Anomaly detection using the simplified Hopfield network proposed in 
        "Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy" <https://openreview.net/forum?id=KkazG4lgKL>.

        For each class, it computes a reference pattern based on the embeddings at correctly classified pixels.
        During inference, it matches the embeddings with the patterns of the predicted class.
        """
        super().__init__()
        self.segmenter = segmenter
        self.hidden_states_index = hidden_states_index
        self.patterns = None


    @torch.no_grad()
    def fit(self, dl_train):
        patterns_accum = torch.zeros((self.segmenter.num_classes, self.segmenter.get_hidden_sizes()[self.hidden_states_index]))
        patterns_count = torch.zeros((self.segmenter.num_classes,))

        self.segmenter.eval()
        for images, labels in tqdm(dl_train):
            labels = labels.cpu()
            segm_logits, hidden_states = self.segmenter(images, return_hidden_states=True)
            embeddings = hidden_states[self.hidden_states_index].permute((0, 2, 3, 1)).cpu() # B x H_l x W_l x emb
            embeddings_shape = (embeddings.shape[1], embeddings.shape[2])
            
            # Match shapes to embedding
            segm_logits = F.interpolate(segm_logits, size=embeddings_shape, mode="bilinear")
            labels = F.interpolate(labels.unsqueeze(1).type(torch.float32), size=(embeddings.shape[1], embeddings.shape[2]), mode="nearest").squeeze(1).type(torch.int32)
            
            # Get correct predictions
            segm_preds = torch.argmax(segm_logits, dim=1).cpu()
            mask_corrects = (segm_preds == labels)
            correct_latents = embeddings[mask_corrects]
            correct_labels = segm_preds[mask_corrects]

            # Keep track of embeddings at correct predictions
            for c in range(self.segmenter.num_classes):
                mask_class = (correct_labels == c)
                patterns_accum[c] += correct_latents[mask_class].sum(dim=0)
                patterns_count[c] += len(correct_latents[mask_class])

        # Average correct embeddings for each class
        self.patterns = patterns_accum / patterns_count.unsqueeze(1).expand_as(patterns_accum)
        self.patterns = torch.nan_to_num(self.patterns)

        return self


    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden_states = self.segmenter(inputs, return_hidden_states=True)
        
        embeddings = hidden_states[self.hidden_states_index].permute((0, 2, 3, 1)).cpu() # B x H_l x W_l x emb
        embedding_shape = (embeddings.shape[1], embeddings.shape[2])
        orig_shape = (inputs.shape[2], inputs.shape[3])
        
        logits_downsampled = F.interpolate(logits, size=embedding_shape, mode="bilinear")
        segm_preds = torch.argmax(logits_downsampled, dim=1).cpu()

        # Compare embeddings with the patterns of the predicted class
        ood_scores = torch.mul(embeddings, self.patterns[segm_preds]).sum(dim=-1)
        ood_scores = F.interpolate(ood_scores.unsqueeze(1), orig_shape, mode="bilinear")

        return logits, ood_scores