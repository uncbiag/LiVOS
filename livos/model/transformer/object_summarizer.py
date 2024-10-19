from typing import List, Dict, Optional, Tuple
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from livos.model.transformer.positional_encoding import PositionalEncoding


# @torch.jit.script
def _weighted_pooling(
    masks: torch.Tensor, value: torch.Tensor, logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # value: B*num_objects*H*W*value_dim
    # logits: B*num_objects*H*W*num_summaries
    # masks: B*num_objects*H*W*num_summaries: 1 if allowed
    weights = logits.sigmoid() * masks
    # B*num_objects*num_summaries*value_dim
    sums_BNQC = torch.einsum('bkhwq,bkhwc->bkqc', weights, value)
    # B*num_objects*H*W*num_summaries -> B*num_objects*num_summaries*1
    area_BNQ1 = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)

    # B*num_objects*num_summaries*value_dim
    return sums_BNQC, area_BNQ1


class ObjectSummarizer(nn.Module):
    def __init__(
        self,
        value_dim: int,
        embed_dim: int,
        num_summaries: int,
        *,
        add_pe: bool=False,
        pixel_pe_scale: int=32,
        pixel_pe_temperature: int=128
    ):
        super().__init__()

        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_summaries = num_summaries
        self.add_pe = add_pe
        self.pixel_pe_scale = pixel_pe_scale
        self.pixel_pe_temperature = pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(
                self.embed_dim, scale=self.pixel_pe_scale, 
                temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def forward(
        self,
        masks_BNHW: torch.Tensor,
        value_BNCHW: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value_BNCHW.shape[-2:]
        masks_BNHW = F.interpolate(masks_BNHW.float(), size=(h, w), mode='area')
        masks_BNHW1 = masks_BNHW.unsqueeze(-1)
        inv_masks_BNHW1 = 1 - masks_BNHW1
        repeated_masks_BNHW2 = torch.cat([
            masks_BNHW1.expand(-1, -1, -1, -1, self.num_summaries // 2),
            inv_masks_BNHW1.expand(-1, -1, -1, -1, self.num_summaries // 2)], dim=-1)

        value_BNHWC = value_BNCHW.permute(0, 1, 3, 4, 2)
        value_BNHWC = self.input_proj(value_BNHWC)
        if self.add_pe:
            pe = self.pos_enc(value_BNHWC)
            value_BNHWC = value_BNHWC + pe

        with torch.amp.autocast('cuda', enabled=False):
            value_BNHWC = value_BNHWC.float()
            feature = self.feature_pred(value_BNHWC)
            logits = self.weights_pred(value_BNHWC)
            sums, area = _weighted_pooling(repeated_masks_BNHW2, feature, logits)

        memory_BNQC = torch.cat([sums, area], dim=-1)

        return memory_BNQC