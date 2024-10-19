import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from livos.model.group_modules import GConv2d
from livos.utils.tensor_utils import aggregate
from livos.model.transformer.positional_encoding import PositionalEncoding
from livos.model.transformer.transformer_layers import *


class QueryTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_queries,
        ff_dim,
        *,
        add_pe_to_qkv: List[bool] = [True, True, False],
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.ff_dim = ff_dim

        self.read_from_pixel = CrossAttention(
            self.embed_dim, self.num_heads, add_pe_to_qkv=add_pe_to_qkv)
        self.self_attn = SelfAttention(
            self.embed_dim, self.num_heads, add_pe_to_qkv=add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(
            self.embed_dim, self.num_heads, add_pe_to_qkv=add_pe_to_qkv, norm=False)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pixel: torch.Tensor,
        query_pe: torch.Tensor,
        pixel_pe: torch.Tensor,
        attn_mask: torch.Tensor,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)

        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C
        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        x, q_weights = self.read_from_pixel(
            x, pixel_flat, query_pe, pixel_pe, attn_mask=attn_mask,
            need_weights=need_weights)

        x = self.self_attn(x, query_pe)
        x = self.ffn(x)

        pixel_flat, p_weights = self.read_from_query(
            pixel_flat, x, pixel_pe, query_pe, need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)

        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            q_weights = q_weights.view(
                bs, num_objects, self.num_heads, self.num_queries, h, w)
            p_weights = p_weights.transpose(2, 3).view(
                bs, num_objects, self.num_heads, self.num_queries, h, w)

        return x, pixel, q_weights, p_weights


class QueryTransformer(nn.Module):
    def __init__(
        self,
        value_dim: int,
        embed_dim: int,
        ff_dim: int,
        num_blocks: int,
        num_heads: int,
        num_queries: int,
        pixel_pe_scale: int=32,
        pixel_pe_temperature: int=128,
    ):
        super().__init__()

        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pixel_pe_scale = pixel_pe_scale
        self.pixel_pe_temperature = pixel_pe_temperature

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)

        # projection from object summaries to query initialization and embedding
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(
            self.embed_dim, scale=self.pixel_pe_scale,
            temperature=self.pixel_pe_temperature, channel_last=False, 
            transpose_output=True)

        # transformer blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(embed_dim, num_heads, num_queries, ff_dim)
            for _ in range(self.num_blocks))
        
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        pixel_BNCHW: torch.Tensor,
        object_memory_BNQC: torch.Tensor,
        selector: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # pixel: B*num_objects*embed_dim*H*W
        # obj_summaries: B*num_objects*T*num_queries*embed_dim
        Q = self.num_queries
        B, N, _, H, W = pixel_BNCHW.shape

        # normalize object values
        # the last channel is the cumulative area of the object
        object_memory_DQC = object_memory_BNQC.view(B * N, Q, self.embed_dim + 1)
        object_sums_DQC = object_memory_DQC[:, :, :-1]
        object_area_DQ1 = object_memory_DQC[:, :, -1:]        
        object_values_DQC = object_sums_DQC / (object_area_DQ1 + 1e-4)
        
        object_init_DQC = self.summary_to_query_init(object_values_DQC)
        object_emb_DQC = self.summary_to_query_emb(object_values_DQC)

        # positional embeddings for object queries
        query_DQC = self.query_init.weight.unsqueeze(0).expand(B * N, -1, -1)
        query_DQC = query_DQC + object_init_DQC

        query_emb_DQC = self.query_emb.weight.unsqueeze(0).expand(B * N, -1, -1)
        query_emb_DQC = query_emb_DQC + object_emb_DQC

        # positional embeddings for pixel features
        pixel_init_BNCHW = self.pixel_init_proj(pixel_BNCHW)
        pixel_emb_BNCHW = self.pixel_emb_proj(pixel_BNCHW)
        pixel_pe = self.spatial_pe(pixel_BNCHW.flatten(0, 1))
        pixel_emb_DLC = pixel_emb_BNCHW.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe_DLC = pixel_pe.flatten(1, 2) + pixel_emb_DLC
        pixel = pixel_init_BNCHW

        # run the transformer
        aux_features = {'q_logits': []}

        # first aux output
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['q_logits'].append(aux_logits)
        for i in range(self.num_blocks):
            query_DQC, pixel, q_weights, p_weights = self.blocks[i](
                query_DQC, pixel, query_emb_DQC, pixel_pe_DLC, attn_mask, 
                need_weights=need_weights)

            if self.training or i <= self.num_blocks - 1 or need_weights:
                aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
                attn_mask = self._get_aux_mask(aux_logits, selector)
                aux_features['q_logits'].append(aux_logits)

        aux_features['q_weights'] = q_weights  # last layer only
        aux_features['p_weights'] = p_weights  # last layer only

        if self.training:
            # no need to save all heads
            aux_features['attn_mask'] = attn_mask.view(
                B, N, self.num_heads, self.num_queries, H, W)[:, :, 0]

        return pixel, aux_features

    def _get_aux_mask(
        self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked

        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)

        is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask
