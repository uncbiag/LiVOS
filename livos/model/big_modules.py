"""
big_modules.py - This file stores higher-level network blocks.

x - usually denotes features that are shared between objects.
g - usually denotes features that are not shared between objects 
    with an extra "num_objects" dimension (batch_size * num_objects * num_channels * H * W).

The trailing number of a variable usually denotes the stride
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from livos.model.group_modules import *
from livos.model.utils import resnet
from livos.model.modules import *


class PixelEncoder(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        self.is_resnet = 'resnet' in model_cfg.pixel_encoder.type
        if self.is_resnet:
            if model_cfg.pixel_encoder.type == 'resnet18':
                network = resnet.resnet18(pretrained=True)
            elif model_cfg.pixel_encoder.type == 'resnet50':
                network = resnet.resnet50(pretrained=True)
            else:
                raise NotImplementedError
            self.conv1 = network.conv1
            self.bn1 = network.bn1
            self.relu = network.relu
            self.maxpool = network.maxpool

            self.res2 = network.layer1
            self.layer2 = network.layer2
            self.layer3 = network.layer3
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f4 = self.res2(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)

        return f16, f8, f4

    # override the default train() to freeze BN statistics
    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class KeyProjection(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        in_dim = model_cfg.pixel_encoder.ms_dims[0]
        mid_dim = model_cfg.pixel_dim
        key_dim = model_cfg.key_dim

        self.pix_feat_proj = nn.Conv2d(in_dim, mid_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(mid_dim, key_dim, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pix_feat_proj(x)
        return self.key_proj(x)


class MaskEncoder(nn.Module):
    def __init__(self, model_cfg: DictConfig, single_object=False):
        super().__init__()
        pixel_dim = model_cfg.pixel_dim
        value_dim = model_cfg.value_dim
        final_dim = model_cfg.mask_encoder.final_dim

        self.single_object = single_object
        extra_dim = 1 if single_object else 2

        if model_cfg.mask_encoder.type == 'resnet18':
            network = resnet.resnet18(pretrained=True, extra_dim=extra_dim)
        elif model_cfg.mask_encoder.type == 'resnet50':
            network = resnet.resnet50(pretrained=True, extra_dim=extra_dim)
        else:
            raise NotImplementedError
        
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool

        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        self.distributor = MainToGroupDistributor()
        self.fuser = GroupFeatureFusionBlock(pixel_dim, final_dim, value_dim)

    def forward(
        self,
        image: torch.Tensor,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        others: torch.Tensor,
        *,
        chunk_size: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ms_features are from the key encoder
        # we only use the first one (lowest resolution), following XMem
        if self.single_object:
            g = masks.unsqueeze(2)
        else:
            g = torch.stack([masks, others], dim=2)

        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
        else:
            fast_path = False

        # chunk-by-chunk inference for object channel
        all_g = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                g_chunk = g
            else:
                g_chunk = g[:, i:i + chunk_size]
            actual_chunk_size = g_chunk.shape[1]
            g_chunk = g_chunk.flatten(start_dim=0, end_dim=1)

            g_chunk = self.conv1(g_chunk)
            g_chunk = self.bn1(g_chunk)  # 1/2, 64
            g_chunk = self.maxpool(g_chunk)  # 1/4, 64
            g_chunk = self.relu(g_chunk)

            g_chunk = self.layer1(g_chunk)  # 1/4
            g_chunk = self.layer2(g_chunk)  # 1/8
            g_chunk = self.layer3(g_chunk)  # 1/16

            g_chunk = g_chunk.view(batch_size, actual_chunk_size, *g_chunk.shape[1:])
            g_chunk = self.fuser(pix_feat, g_chunk)
            all_g.append(g_chunk)
        g = torch.cat(all_g, dim=1)

        return g

    # override the default train() to freeze BN statistics
    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class PixelFeatureFuser(nn.Module):
    def __init__(self, model_cfg: DictConfig, single_object=False):
        super().__init__()
        value_dim = model_cfg.value_dim
        pixel_dim = model_cfg.pixel_dim
        embed_dim = model_cfg.embed_dim
        self.single_object = single_object

        self.fuser = GroupFeatureFusionBlock(pixel_dim, value_dim, embed_dim)

    def forward(
        self,
        pix_feat: torch.Tensor,
        pixel_memory: torch.Tensor,
        last_mask: torch.Tensor,
        last_others: torch.Tensor,
        *,
        chunk_size: int = -1
    ) -> torch.Tensor:
        batch_size, num_objects = pixel_memory.shape[:2]

        if self.single_object:
            last_mask = last_mask.unsqueeze(2)
        else:
            last_mask = torch.stack([last_mask, last_others], dim=2)

        if chunk_size < 1:
            chunk_size = num_objects

        # chunk-by-chunk inference
        all_p16 = []
        for i in range(0, num_objects, chunk_size):
            p16 = pixel_memory[:, i:i + chunk_size]
            p16 = self.fuser(pix_feat, p16)
            all_p16.append(p16)
        p16 = torch.cat(all_p16, dim=1)

        return p16


class MaskDecoder(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        embed_dim = model_cfg.embed_dim
        ms_image_dims = model_cfg.pixel_encoder.ms_dims
        up_dims = model_cfg.mask_decoder.up_dims

        assert embed_dim == up_dims[0]

        self.decoder_feat_proc = \
            DecoderFeatureProcessor(ms_image_dims[1:], up_dims[:-1])
        self.up_16_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.up_8_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])

        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=3, padding=1)

    def forward(
        self,
        ms_image_feat: Iterable[torch.Tensor],
        memory_readout: torch.Tensor,
        *,
        chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, num_objects = memory_readout.shape[:2]
        f8, f4 = self.decoder_feat_proc(ms_image_feat[1:])
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
        else:
            fast_path = False

        # chunk-by-chunk inference
        all_logits = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                p16 = memory_readout
            else:
                p16 = memory_readout[:, i:i + chunk_size]
            C = p16.shape[1] # actual chunk size

            p8 = self.up_16_8(p16, f8)
            p4 = self.up_8_4(p8, f4)
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.pred(F.relu(p4.flatten(start_dim=0, end_dim=1).float()))

            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        logits = logits.view(B, num_objects, *logits.shape[-2:])

        return logits
