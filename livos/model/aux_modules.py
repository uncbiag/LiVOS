"""
For computing auxiliary outputs for auxiliary losses
"""
import torch
import torch.nn as nn
from typing import Dict

from livos.model.group_modules import GConv2d
from livos.utils.tensor_utils import aggregate


class LinearPredictor(nn.Module):
    def __init__(self, x_dim: int, pix_dim: int):
        super().__init__()
        self.projection = GConv2d(x_dim, pix_dim + 1, kernel_size=1)

    def forward(self, pixfeat: torch.Tensor, sensory: torch.Tensor) -> torch.Tensor:
        # pixel_feat: B*pix_dim*H*W
        # x: B*num_objects*x_dim*H*W
        num_objects = sensory.shape[1]
        sensory = self.projection(sensory.clone())

        pixfeat = pixfeat.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        logits = (pixfeat * sensory[:, :, :-1]).sum(dim=2) + sensory[:, :, -1]
        return logits


class DirectPredictor(nn.Module):
    def __init__(self, x_dim: int):
        super().__init__()
        self.projection = GConv2d(x_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B*num_objects*x_dim*H*W
        logits = self.projection(x).squeeze(2)
        return logits


class AuxComputer(nn.Module):
    def __init__(self, sensory_dim, embed_dim):
        super().__init__()

        self.sensory_aux = LinearPredictor(sensory_dim, embed_dim)

    def aggregate_with_selector(
        self, logits: torch.Tensor, selector: torch.Tensor
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        logits = aggregate(prob, dim=1)
        return logits

    def forward(
        self, 
        pixfeat: torch.Tensor,
        sensory: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        *,
        selector: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        aux_output = {}
        aux_output['attn_mask'] = aux['attn_mask']

        # B*num_objects*H*W
        logits = self.sensory_aux(pixfeat, sensory)
        aux_output['sensory_logits'] = self.aggregate_with_selector(logits, selector)

        q_logits = aux['q_logits']
        # B*num_objects*num_levels*H*W
        aux_output['q_logits'] = self.aggregate_with_selector(
            torch.stack(q_logits, dim=2),
            selector.unsqueeze(2) if selector is not None else None)

        return aux_output