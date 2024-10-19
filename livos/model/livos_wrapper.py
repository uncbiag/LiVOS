import logging
import math
import numpy as np
from typing import Tuple, Iterable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from livos.model.utils import resnet
from livos.model.aux_modules import AuxComputer
from livos.model.transformer.object_transformer import QueryTransformer
from livos.model.transformer.object_summarizer import ObjectSummarizer
from livos.utils.tensor_utils import aggregate

log = logging.getLogger()


class KeyProj(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.layer1(x))
        return x


class PixProj(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()        
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class GateProj(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x_BCHW: torch.Tensor) -> torch.Tensor:
        x_BC = torch.mean(torch.sigmoid(self.layer(x_BCHW)), dim=[2,3])
        return x_BC


class CAResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

        t = int((abs(math.log2(out_dim)) + 1) // 2)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        if self.residual:
            if in_dim == out_dim:
                self.downsample = nn.Identity()
            else:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))

        b, c = x.shape[:2]
        w = self.pool(x).view(b, 1, c)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1).sigmoid()  # B*C*1*1

        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w

        return x


class FeatureFuser(nn.Module):
    def __init__(self, input_x_dim: int, input_y_dim: int, out_dim: int):
        super().__init__()
        self.x_transform = nn.Conv2d(input_x_dim, out_dim, kernel_size=1)
        self.y_transform = nn.Conv2d(input_y_dim, out_dim, kernel_size=1)

        self.layer1 = CAResBlock(out_dim, out_dim)
        self.layer2 = CAResBlock(out_dim, out_dim)

    def forward(self, x_BCHW: torch.Tensor, y_BNCHW: torch.Tensor) -> torch.Tensor:
        """ Fuse pixel and value features 
        Args:
            x_BCHW: pixel feature 
            y_BNCHW: value feature
        """
        B, N = y_BNCHW.shape[:2]

        x_BCHW = self.x_transform(x_BCHW)
        x_BNCHW = x_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        x_DCHW = x_BNCHW.flatten(start_dim=0, end_dim=1)
 
        y_DCHW = y_BNCHW.flatten(start_dim=0, end_dim=1)
        y_DCHW = self.y_transform(y_DCHW)
 
        # Add x and y
        fused_DCHW = x_DCHW + y_DCHW
        fused_DCHW = self.layer2(self.layer1(fused_DCHW))
        fused_BNCHW = fused_DCHW.view(B, N, *fused_DCHW.shape[1:])

        return fused_BNCHW


class PixelFuser(nn.Module):
    """ Fuse pixel feature with value readout, as well as sensory.
    """
    def __init__(self, pix_dim, embed_dim, value_dim, sensory_dim):
        super().__init__()
        self.fuser = FeatureFuser(pix_dim, embed_dim, value_dim)
        self.conv = nn.Conv2d(sensory_dim + 2, value_dim, kernel_size=1)

    def forward(
        self, 
        pixfeat_BCHW: torch.Tensor, 
        readout_BNCHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        masks_BNHW: torch.Tensor,
        masks_others_BNHW: torch.Tensor,
        *,
        chunk_size: int=-1,
    ) -> torch.Tensor:      
        N = masks_BNHW.shape[1]        
        chunk_size = N if chunk_size < 0 else chunk_size
        masks_BN2HW = torch.stack([masks_BNHW, masks_others_BNHW], dim=2)

        # chunk-by-chunk inference
        X_all = []
        for i in range(0, N, chunk_size):
            sensory_chunk_BNCHW = sensory_BNCHW[:, i:i+chunk_size]
            masks_chunk_BN2HW = masks_BN2HW[:, i:i+chunk_size]
            sensory_chunk_BNCHW = torch.cat(
                [sensory_chunk_BNCHW, masks_chunk_BN2HW], dim=2)

            B, M = sensory_chunk_BNCHW.shape[:2]
            sensory_chunk_DCHW = sensory_chunk_BNCHW.flatten(start_dim=0, end_dim=1)
            sensory_chunk_DCHW = self.conv(sensory_chunk_DCHW)
            sensory_chunk_BNCHW = \
                sensory_chunk_DCHW.view(B, M, *sensory_chunk_DCHW.shape[1:])

            X_BNCHW = readout_BNCHW[:, i:i + chunk_size] + sensory_chunk_BNCHW
            X_BNCHW = self.fuser(pixfeat_BCHW, X_BNCHW)
            X_all.append(X_BNCHW)
        fused = torch.cat(X_all, dim=1)

        return fused


def recurrent_update(
    h_BNCHW: torch.Tensor, 
    values_BNCHW: torch.Tensor
) -> torch.Tensor:
    """
    """
    dim = values_BNCHW.shape[2] // 3
    forget_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, :dim])
    update_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, dim:dim*2])
    new_values_BNCHW = torch.tanh(values_BNCHW[:, :, dim*2:])
    new_h_BNCHW = forget_gate_BNCHW * h_BNCHW * (1. - update_gate_BNCHW) + \
        update_gate_BNCHW * new_values_BNCHW

    return new_h_BNCHW


class MultiscaleSensoryUpdater(nn.Module):
    """ Used in the mask decoder with multi-scale feature and GRU.
    """
    def __init__(self, g_dims: List[int], mid_dim: int, sensory_dim: int):
        super().__init__()
        self.g16_conv = nn.Conv2d(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = nn.Conv2d(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = nn.Conv2d(g_dims[2]+1, mid_dim, kernel_size=1)
        self.transform = nn.Conv2d(
            mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def downsample(
        self, 
        input_BCHW: torch.Tensor, 
        ratio: float, 
        mode: str='area', 
        align_corners: bool=None
    ) -> torch.Tensor:
        """
        """
        output_BCHW = F.interpolate(
            input_BCHW, scale_factor=ratio, mode=mode, align_corners=align_corners)
        return output_BCHW

    def forward(self, g: torch.Tensor, h_BNCHW: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: multiscale features
            h_BNCHW: sensory 
        
        Returns:
            None
        """
        B, N = g[0].shape[:2]
        g16_DCHW, g8_DCHW, g4_DCHW = [x.flatten(start_dim=0, end_dim=1) for x in g]

        # Muti-scale fusion
        g_DCHW = self.g16_conv(g16_DCHW) + \
            self.g8_conv(self.downsample(g8_DCHW, ratio=1/2)) + \
            self.g4_conv(self.downsample(g4_DCHW, ratio=1/4))

        h_DCHW = h_BNCHW.flatten(start_dim=0, end_dim=1)
        values_DCHW = self.transform(torch.cat([g_DCHW, h_DCHW], dim=1))
        values_BNCHW = values_DCHW.view(B, N, *values_DCHW.shape[1:])
        new_h_BNCHW = recurrent_update(h_BNCHW.clone(), values_BNCHW)

        return new_h_BNCHW


class SensoryUpdater(nn.Module):
    """ Used in the mask encoder for deep update.
    """
    def __init__(self, value_dim: int, sensory_dim: int):
        super().__init__()
        self.transform = nn.Conv2d(
            value_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def recurrent_update(
        self, h_BNCHW: torch.Tensor, values_BNCHW: torch.Tensor
    ) -> torch.Tensor:
        dim = values_BNCHW.shape[2] // 3
        forget_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, :dim])
        update_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, dim:dim*2])
        new_values_BNCHW = torch.tanh(values_BNCHW[:, :, dim*2:])
        new_h_BNCHW = forget_gate_BNCHW * h_BNCHW * (1. - update_gate_BNCHW) + \
            update_gate_BNCHW * new_values_BNCHW

        return new_h_BNCHW

    def forward(self, g_BNCHW: torch.Tensor, h_BNCHW: torch.Tensor) -> torch.Tensor:
        B, N = g_BNCHW.shape[:2]
        gh_BNCHW = torch.cat([g_BNCHW, h_BNCHW], dim=2)
        gh_DCHW = gh_BNCHW.flatten(start_dim=0, end_dim=1)
        values_DCHW = self.transform(gh_DCHW)
        values_BNCHW = values_DCHW.view(B, N, *values_DCHW.shape[1:])
        new_h_BNCHW = recurrent_update(h_BNCHW.clone(), values_BNCHW)

        return new_h_BNCHW


class ImageEncoder(nn.Module):
    def __init__(self, encoder_type: str='resnet50'):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=True)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=True)
        else:
            raise NotImplementedError
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (tesnor): shape of B, C, H, W
        Returns:        
            f16 (tensor): shape of B, C, H/16, w/16. C={1024, 256} for resnet{50, 18}.
            f8  (tesnor): shape of B, C, H/8, w/8. C={512, 128} for resnet{50, 18}.
            f4  (tensor): shape of B, C, H/4, w/4. C={256, 64} for resnet{50, 18}.
        """
        f2 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f4 = self.layer1(f2)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)

        return f16, f8, f4


class MaskEncoder(nn.Module):
    def __init__(
        self, 
        pix_dim: int, 
        value_dim: int, 
        sensory_dim: int, 
        encoder_type: str='resnet18'
    ):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=True, extra_dim=2)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=True, extra_dim=2)
        else:
            raise NotImplementedError
        
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        embed_dim = {'resnet18': 256, 'resnet50': 1024}[encoder_type]
        self.fuser = FeatureFuser(pix_dim, embed_dim, value_dim)
        self.sensory_updater = SensoryUpdater(value_dim, sensory_dim)

    def forward(
        self, 
        image_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        masks_BN2HW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        deep_update: bool = False,
        chunk_size=-1
    ) -> torch.Tensor:
        B, N, K = masks_BN2HW.shape[:3]
        assert K == 2

        image_BN3HW = image_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1) # C = 3
        input_BN5HW = torch.cat([image_BN3HW, masks_BN2HW], dim=2)
        chunk_size = N if chunk_size < 1 or chunk_size > N else chunk_size

        new_sensory_BNCHW = sensory_BNCHW
        if deep_update and chunk_size != N:
            new_sensory_BNCHW = torch.empty_like(sensory_BNCHW)

        X_chunks = []
        for i in range(0, N, chunk_size):
            X_BM5HW = input_BN5HW[:, i:i+chunk_size] # M = chunk size
            X_D5HW = X_BM5HW.flatten(start_dim=0, end_dim=1) # D=B*N
            X_DCHW = self.maxpool(self.relu(self.bn1(self.conv1(X_D5HW))))
            X_DCHW = self.layer1(X_DCHW) # 1/4
            X_DCHW = self.layer2(X_DCHW) # 1/8
            X_DCHW = self.layer3(X_DCHW) # 1/16
            X_BMCHW = X_DCHW.view(B, X_BM5HW.shape[1], *X_DCHW.shape[1:])
            X_BMCHW = self.fuser(pixfeat_BCHW, X_BMCHW)
            X_chunks.append(X_BMCHW)

            if deep_update:
                sensory_chunk_BMCHW = sensory_BNCHW[:, i:i+chunk_size]
                new_sensory_BNCHW[:, i:i+chunk_size] = \
                    self.sensory_updater(X_BMCHW, sensory_chunk_BMCHW)

        X_BNCHW = torch.cat(X_chunks, dim=1)

        return X_BNCHW, new_sensory_BNCHW


class MaskUpsampleBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        if in_dim == out_dim:
            self.linear_proj = nn.Identity()
        else:
            self.linear_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        self.scale_factor = scale_factor

    def upsample(
        self, feat: torch.Tensor, ratio: float, 
        mode: str = 'bilinear', align_corners: bool = False
    ) -> torch.Tensor:
        B, N = feat.shape[:2]
        feat = F.interpolate(
            feat.flatten(start_dim=0, end_dim=1), 
            scale_factor=ratio, mode=mode, align_corners=align_corners)
        feat = feat.view(B, N, *feat.shape[1:])
        return feat

    def forward(self, feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (tensor): B,N,C,H,W
            skip_feat (tensor): B,C,2H,2W
        """
        B, N = feat.shape[:2]
        feat_BNCHW, skip_BCHW = feat, skip_feat
        feat_BNCHW = self.upsample(feat_BNCHW, ratio=self.scale_factor)
        skip_BNCHW = skip_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        feat_BNCHW = feat_BNCHW + skip_BNCHW
        feat_DCHW = feat_BNCHW.flatten(start_dim=0, end_dim=1)
        feat_DCHW = self.linear_proj(feat_DCHW) + \
            self.act2(self.conv2(self.act1(self.conv1(feat_DCHW))))
        feat_BNCHW = feat_DCHW.view(B, N, *feat_DCHW.shape[1:])

        return feat_BNCHW


class MaskDecoder(nn.Module):
    def __init__(self, ms_dims, up_dims, sensory_dim):
        super().__init__()
        self.linear_proj_f8 = nn.Conv2d(ms_dims[1], up_dims[0], kernel_size=1)
        self.linear_proj_f4 = nn.Conv2d(ms_dims[2], up_dims[1], kernel_size=1)
        self.upsample_16_to_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.upsample_8_to_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])
        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=1)

        self.sensory_updater = MultiscaleSensoryUpdater(
            up_dims, sensory_dim, sensory_dim)

    def forward(
        self, 
        ms_feats: Iterable[torch.Tensor],
        readout_BNCHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        update_sensory: bool=False,
        chunk_size: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = readout_BNCHW.shape[:2]
        v16_BNCHW = readout_BNCHW
        f8_BCHW, f4_BCHW = ms_feats[1:] # why ignoring f16?
        f8_BCHW = self.linear_proj_f8(f8_BCHW)
        f4_BCHW = self.linear_proj_f4(f4_BCHW)

        chunk_size = N if chunk_size < 1 or chunk_size > N else chunk_size
        new_sensory_BNCHW = sensory_BNCHW
        if update_sensory and chunk_size != N:
            new_sensory_BNCHW = torch.empty_like(sensory_BNCHW)

        logits_chunks = []
        for i in range(0, N, chunk_size):
            v16_chunk_BMCHW = v16_BNCHW[:, i:i+chunk_size]
            M = v16_chunk_BMCHW.shape[1] # actual chunk size
            v8_chunk_BMCHW = self.upsample_16_to_8(v16_chunk_BMCHW, f8_BCHW)
            v4_chunk_BMCHW = self.upsample_8_to_4(v8_chunk_BMCHW, f4_BCHW)
            v4_chunk_DCHW = v4_chunk_BMCHW.flatten(start_dim=0, end_dim=1)
            logits_chunk_D1HW = self.pred(v4_chunk_DCHW) # TODO: Add F.relu()
            logits_chunk_BMHW = \
                logits_chunk_D1HW.view(B, M, *logits_chunk_D1HW.shape[-2:])

            if update_sensory:
                logits_chunk_BM1HW = logits_chunk_BMHW.unsqueeze(2)
                v4_chunk_BMCHW = torch.cat([v4_chunk_BMCHW, logits_chunk_BM1HW], dim=2)
                new_sensory_BNCHW[:, i:i+chunk_size] = self.sensory_updater(
                    [v16_chunk_BMCHW, v8_chunk_BMCHW, v4_chunk_BMCHW], 
                    sensory_BNCHW[:, i:i+chunk_size])

            logits_chunks.append(logits_chunk_BMHW)
        logits_BNCHW = torch.cat(logits_chunks, dim=1)

        return logits_BNCHW, new_sensory_BNCHW


class LIVOS(nn.Module):
    def __init__(
            self,
            model_type='base',
            image_encoder_type='resnet50',
            mask_encoder_type='resnet18',
    ) -> None:
        super().__init__()
        self.ms_dims = {
            'resnet50': [1024, 512, 256], 
            'resnet18': [256, 128, 64],
        }[image_encoder_type]

        self.up_dims = {
            'base': [256, 128, 128],
            'small': [256, 128, 64]
        }[model_type]

        self.key_dim = 64
        self.value_dim = 256
        self.pixel_dim = 256
        self.sensory_dim = 256
        self.embed_dim = 256

        # Parameters for the query transformer
        self.num_blocks = 3
        self.num_heads = 8
        self.num_queries = 16
        self.ff_dim = 2048

        self.image_encoder = ImageEncoder(encoder_type=image_encoder_type)
        self.mask_encoder = MaskEncoder(
            self.pixel_dim, self.value_dim, self.sensory_dim, 
            encoder_type=mask_encoder_type)
        
        self.key_projector = KeyProj(self.ms_dims[0], self.pixel_dim, self.key_dim)
        self.pix_projector = PixProj(self.ms_dims[0], self.pixel_dim)
        self.gate_projector = GateProj(self.ms_dims[0], self.value_dim)        
        self.mask_decoder = MaskDecoder(self.ms_dims, self.up_dims, self.sensory_dim)
        self.pixel_fuser = PixelFuser(
            self.pixel_dim, self.embed_dim, self.value_dim, self.sensory_dim)

        self.object_transformer = QueryTransformer(
            self.value_dim, self.embed_dim, self.ff_dim, self.num_blocks,
            self.num_heads, self.num_queries)

        self.object_summarizer = ObjectSummarizer(
            self.value_dim, self.embed_dim, self.num_queries, add_pe=True)

        self.aux_computer = AuxComputer(self.sensory_dim, self.embed_dim)

        self.register_buffer(
            "pixel_mean", torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), False)
        self.register_buffer(
            "pixel_std", torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), False)

    def encode_image(
        self, image: torch.Tensor
    ) -> Tuple[Iterable[torch.Tensor], torch.Tensor]:
        
        image = (image - self.pixel_mean) / self.pixel_std
        multiscale_feats = self.image_encoder(image)
        return multiscale_feats
    
    def encode_mask(
        self, 
        image_BCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor, 
        mask_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        deep_update: bool=False,
        chunk_size: int=-1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image (tensor): B, C, H, W.
            pixfeat (tensor): B, C, H, W.
            mask (tensor): B, N, H, W. No background in the mask.
        Returns:
            value (tensor): shape of BNCHW.
        """
        image_BCHW = (image_BCHW - self.pixel_mean) / self.pixel_std
        mask_other_BNHW = (mask_BNHW.sum(dim=1, keepdim=True) - mask_BNHW).clamp(0, 1)
        masks_BN2HW = torch.stack([mask_BNHW, mask_other_BNHW], dim=2)
        mask_value_BNCHW, sensory_BNCHW = self.mask_encoder(
            image_BCHW, pixfeat_BCHW, masks_BN2HW, sensory_BNCHW, 
            deep_update=deep_update, chunk_size=chunk_size)

        object_memory_BNQC = self.object_summarizer(mask_BNHW, mask_value_BNCHW)

        return mask_value_BNCHW, sensory_BNCHW, object_memory_BNQC

    def pixel_fusion(
        self, 
        pixfeat_BCHW: torch.Tensor, 
        readout_BNCHW: torch.Tensor, 
        sensory_BNCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
    ) -> torch.Tensor:
        masks_BNHW = F.interpolate(
            last_masks_BNHW.float(), size=readout_BNCHW.shape[-2:], mode='area')
        masks_sum_B1HW = masks_BNHW.sum(dim=1, keepdim=True)
        masks_others_BNHW = (masks_sum_B1HW - masks_BNHW).clamp(0, 1)

        readout_fused_BNCHW = self.pixel_fuser(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, masks_BNHW, masks_others_BNHW)

        return readout_fused_BNCHW

    def segment(
        self, 
        ms_feats: Tuple[torch.Tensor], 
        readout_BNCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        object_memory_BNQC: torch.Tensor,
        *,
        selector: Optional[torch.Tensor] = None,
        update_sensory: bool=False,
        chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ms_feats: multi-scale features extracted from image
            value (tensor): mask value
        """
        readout_BNCHW = self.pixel_fusion(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, last_masks_BNHW)

        # Enhance the readout with object memory
        readout_BNCHW, aux_features = self.object_transformer(
            readout_BNCHW, object_memory_BNQC, selector=selector)

        logits, sensory_BNCHW = self.mask_decoder(
            ms_feats, readout_BNCHW, sensory_BNCHW, 
            chunk_size=chunk_size, update_sensory=update_sensory)
        
        masks = torch.sigmoid(logits)
        if selector is not None:
            masks = masks * selector

        # Softmax over all objects
        logits = aggregate(masks, dim=1)
        logits = F.interpolate(
            logits, scale_factor=4, mode='bilinear', align_corners=False)
        masks = F.softmax(logits, dim=1)

        return logits, masks, sensory_BNCHW, aux_features

    def forward(self, data: Dict):
        """
        Args:
            H, W: image spatial dimensions.                        
        """
        out = {}
        num_objects = [num.item() for num in data['info']['num_objects']]
        out['num_objects'] = num_objects

        max_num_objects = max(num_objects) # actual max num of objects

        images_BTCHW = data['rgb']
        B, T = images_BTCHW.shape[:2]
        images_DCHW = images_BTCHW.view(B*T, *images_BTCHW.shape[2:])

        ms_feats = self.encode_image(images_DCHW) # list of tensors with shape D,C,H,W
        feat_DCHW = ms_feats[0]
        pixfeat_DCHW = self.pix_projector(feat_DCHW)
        pixfeat_BTCHW = pixfeat_DCHW.view(B, T, *pixfeat_DCHW.shape[1:]) # 1/16

        key_DCHW = self.key_projector(feat_DCHW)
        key_BTCHW = key_DCHW.view(B, T, *key_DCHW.shape[1:]) # 1/16

        gate_DC = self.gate_projector(feat_DCHW)
        gate_BTC = gate_DC.view(B, T, *gate_DC.shape[1:])

        first_frame_image_BCHW = images_BTCHW[:, 0]
        first_frame_pixfeat_BCHW = pixfeat_BTCHW[:, 0]

        first_frame_mask_B1MHW = data['first_frame_gt']
        first_frame_mask_BNHW = first_frame_mask_B1MHW[:, 0, :max_num_objects]

        # Initialize sensory
        N, Cs = max_num_objects, self.sensory_dim, 
        H, W = key_DCHW.shape[2:]
        sensory_BNCHW = torch.zeros(B, N, Cs, H, W, device=key_DCHW.device)

        # Encode mask
        value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
            first_frame_image_BCHW, first_frame_pixfeat_BCHW, 
            first_frame_mask_BNHW, sensory_BNCHW, deep_update=True)

        # Object memory summation
        object_memory_sum_BNQC = object_memory_BNQC

        # Obtain the state matrix
        key_BCHW = key_BTCHW[:, 0]
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)
        state_BNCC = torch.einsum(
            'bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW).contiguous()

        last_masks_BNHW = first_frame_mask_BNHW
        key_sum_BCHW = key_BCHW
        for i in range(1, T):
            this_key_BCHW = key_BTCHW[:, i]

            # Softmax attention for the key
            this_key_max_B1HW = torch.max(this_key_BCHW, dim=1, keepdim=True).values
            this_key_BCHW = (this_key_BCHW - this_key_max_B1HW).softmax(dim=1)

            this_readout_BNCHW = torch.einsum(
                'bkhw,bnkv->bnvhw', this_key_BCHW, state_BNCC.clone()).contiguous()

            normalizer_B = torch.einsum('bkhw,bkhw->b', this_key_BCHW, key_sum_BCHW)
            this_readout_BNCHW = this_readout_BNCHW / normalizer_B.view(B, 1, 1, 1, 1)
            key_sum_BCHW = key_sum_BCHW + this_key_BCHW

            # Segment
            this_pixfeat_BCHW = pixfeat_BTCHW[:, i]
            this_ms_feats = [feat_DCHW.view(B, T, *feat_DCHW.shape[1:])[:, i] \
                             for feat_DCHW in ms_feats]
            
            this_logits_BNHW, this_masks_BNHW, sensory_BNCHW, aux = self.segment(
                this_ms_feats, this_readout_BNCHW, this_pixfeat_BCHW, last_masks_BNHW, 
                sensory_BNCHW, object_memory_sum_BNQC, update_sensory=True)

            # Auxilary task
            this_aux_output = self.aux_computer(this_pixfeat_BCHW, sensory_BNCHW, aux)

            # Update last masks
            last_masks_BNHW = this_masks_BNHW[:, 1:] # remove the background

            # No need to encode the last frame
            if i < T - 1:
                this_image_BCHW = images_BTCHW[:, i]
                this_value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
                    this_image_BCHW, this_pixfeat_BCHW, this_masks_BNHW[:, 1:], 
                    sensory_BNCHW, deep_update=True)

                # Update the state matrix with gated selection
                this_state_BNCC = torch.einsum(
                    'bkhw,bnvhw->bnkv', this_key_BCHW, this_value_BNCHW).contiguous()
                
                this_gate_BCC = torch.diag_embed(gate_BTC[:, i])
                state_gated_BNCC = torch.einsum(
                    'bnkv,bvv->bnkv', state_BNCC, this_gate_BCC)
                state_BNCC = state_gated_BNCC + this_state_BNCC

                # Update the object memory sum
                object_memory_sum_BNQC = object_memory_sum_BNQC + object_memory_BNQC

            out[f'logits_{i}'] = this_logits_BNHW
            out[f'masks_{i}'] = this_masks_BNHW[:, 1:] # remove the background
            out[f'aux_{i}'] = this_aux_output

        return out
    
    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        """
        If the model is multiple-object and we are training in single-object, 
        we strip the last channel of conv1.
        This is not supposed to happen in standard training except when users are trying to
        finetune a trained model with single object datasets.
        """
        # Map single-object weight to multi-object weight (4->5 out channels in conv1)
        for k in list(src_dict.keys()):
            if k == 'mask_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
            elif k == 'pixel_fuser.sensory_compress.weight':
                if src_dict[k].shape[1] == self.sensory_dim + 1:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')

        self.load_state_dict(src_dict, strict=False)
