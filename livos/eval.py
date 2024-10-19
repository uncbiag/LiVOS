import argparse
from os import path
import hydra
from hydra.core.hydra_config import HydraConfig
import os
from omegaconf import DictConfig
import shutil
import torch

from typing import List, Optional, Iterable, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from livos.inference.data.vos_test_dataset import VOSTestDataset
from livos.inference.data.burst_test_dataset import BURSTTestDataset
from livos.inference.object_manager import ObjectManager
from livos.inference.utils.results_utils import ResultSaver, make_zip
from livos.inference.utils.burst_utils import BURSTResultHandler
from livos.inference.utils.args_utils import get_dataset_cfg
from livos.model.livos_wrapper import LIVOS
from livos.utils.tensor_utils import pad_divide_by, unpad, aggregate


class InferenceCore:
    def __init__(self, network: LIVOS) -> None:
        self.network = network
        self.current_frame_idx = -1
        self.state_BNCC = None
        self.key_sum_BCHW = None
        self.sensory_BNCHW = None
        self.obj_mem_sum_BNQC = None
        self.last_masks_BNHW = None
        self.object_manager = ObjectManager()

    def step(
        self, 
        image: torch.Tensor, 
        mask_HW: Optional[torch.Tensor] = None,
        objects: Optional[List[int]] = None, 
        is_last_frame: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            image (tensor): shape C,H,W
            mask (tensor): shape 
        """
        self.current_frame_idx += 1
        image_CHW = image
        image_CHW, pad_P = pad_divide_by(image_CHW, 16) # P = 4
        image_BCHW = image_CHW.unsqueeze(0) # B = 1

        # Enocde image.
        ms_feats = self.network.encode_image(image_BCHW) # list of [B,C,H,W] tensors
        feat_BCHW = ms_feats[0] # 1/16
        pixfeat_BCHW = self.network.pix_projector(feat_BCHW) # 1/16
        key_BCHW = self.network.key_projector(feat_BCHW) # 1/16        
        gate_BC = self.network.gate_projector(feat_BCHW) # 1/16

        # Normalize the key.
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)

        B, _, H, W = key_BCHW.shape
        if self.current_frame_idx == 0:
            # The first frame must have mask and valid object(s).
            assert objects is not None and mask_HW is not None

            # A temp id indicates the position of an object in the mask tensor.
            # object ids start from 1 due to the background.
            temp_ids, _ = self.object_manager.add_new_objects(objects)
            mask_HW, _ = pad_divide_by(mask_HW, 16)
            mask_NHW = torch.stack(
                [mask_HW == objects[i] for i in range(len(temp_ids))], dim=0).float()            
            prob_with_bg_NHW = aggregate(mask_NHW, dim=0)

            # Initialize sensory.
            N = mask_NHW.shape[0]
            C = self.network.sensory_dim
            sensory_BNCHW = torch.zeros(B, N, C, H, W, device=image.device)

            # Encode mask.
            mask_BNHW = mask_NHW.unsqueeze(0)
            value_BNCHW, sensory_BNCHW, obj_mem_BNQC = self.network.encode_mask(
                image_BCHW, pixfeat_BCHW, mask_BNHW, sensory_BNCHW, deep_update=True)
            
            # Initialize the object memory sum
            self.obj_mem_sum_BNQC = obj_mem_BNQC

            # Get the initial state.
            self.state_BNCC = torch.einsum('bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)
            
            self.key_sum_BCHW = key_BCHW
            self.sensory_BNCHW = sensory_BNCHW
            self.last_masks_BNHW = mask_BNHW

            # Record object states in case of objects appearing in intermediate frames.
            self.object_states = [(0, N, 0)] # (start obj id, end obj id, key_sum)
        else:
            # Get the value for the query frame.
            readout_BNCHW = torch.einsum(
                'bkhw,bnkv->bnvhw', key_BCHW, self.state_BNCC)
            norm_B = torch.einsum('bchw,bchw->b', key_BCHW, self.key_sum_BCHW)
            norm_B = norm_B.view(B, 1, 1, 1, 1)

            # Normalization for query readout.
            for i, j, prev_key_sum in self.object_states:
                if isinstance(prev_key_sum, int):
                    readout_BNCHW[:, i:j] /= norm_B
                else:
                    prev_norm_B = torch.einsum('bkhw,bkhw->b', key_BCHW, prev_key_sum)
                    readout_BNCHW[:, i:j] /= (norm_B - prev_norm_B)

            _, prob_with_bg_BNHW, self.sensory_BNCHW, _ = self.network.segment(
                ms_feats, readout_BNCHW, pixfeat_BCHW, self.last_masks_BNHW, 
                self.sensory_BNCHW, self.obj_mem_sum_BNQC, update_sensory=True)

            # Intermediate frames may also have new object(s), e.g., in YouTube 2019.
            # If so, we need to insert the new objects back into the segmentation.
            new_objects = objects is not None and mask_HW is not None
            if new_objects:
                temp_ids, _ = self.object_manager.add_new_objects(objects)
                mask_HW, _ = pad_divide_by(mask_HW, 16)
                prob_with_bg_BNHW[:, :, mask_HW > 0] = 0 # mutual exclusive

                new_masks = []
                for idx, temp_id in enumerate(temp_ids):
                    this_mask = (mask_HW == objects[idx]).type_as(prob_with_bg_BNHW)
                    new_masks.append(this_mask.unsqueeze(0).unsqueeze(0))
                prob_with_bg_BNHW = torch.cat([prob_with_bg_BNHW, *new_masks], dim=1)

                # Expand the sensory.
                K, C = len(temp_ids), self.network.sensory_dim
                sensory_BKCHW = torch.zeros(B, K, C, H, W, device=image.device)
                self.sensory_BNCHW = torch.cat(
                    [self.sensory_BNCHW, sensory_BKCHW], dim=1)

                # Expand the state matrix.
                state_BKCC = torch.zeros(
                    B, K, *self.state_BNCC.shape[2:], device=image.device)
                self.state_BNCC = torch.cat([self.state_BNCC, state_BKCC], dim=1)

                # Expand the object mem.
                obj_mem_sum_BKQC = torch.zeros(
                    B, K, *self.obj_mem_sum_BNQC.shape[2:], device=image.device)
                self.obj_mem_sum_BNQC = torch.cat(
                    [self.obj_mem_sum_BNQC, obj_mem_sum_BKQC], dim=1)

                # Record the starting point when normaling new objects.
                N = self.state_BNCC.shape[1]
                self.object_states.append((N-K, N, self.key_sum_BCHW))

            # Update the key sum.
            self.key_sum_BCHW = self.key_sum_BCHW + key_BCHW

            # Output probability map.
            prob_with_bg_NHW = prob_with_bg_BNHW[0]

            # Encode the mask to obtain new value.
            mask_BNHW = prob_with_bg_BNHW[:, 1:]
            value_BNCHW, self.sensory_BNCHW, obj_mem_BNQC = self.network.encode_mask(
                image_BCHW, pixfeat_BCHW, mask_BNHW, self.sensory_BNCHW, 
                deep_update=True)

            # Update state with a gate.
            if not is_last_frame:
                self.state_BNCC = torch.einsum(
                    'bvv,bnkv->bnkv', torch.diag_embed(gate_BC), self.state_BNCC)
                this_state_BNCC = torch.einsum(
                    'bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)
                self.state_BNCC += this_state_BNCC

            # Update last masks.
            self.last_masks_BNHW = mask_BNHW

            # Update the object memory sum
            self.obj_mem_sum_BNQC = self.obj_mem_sum_BNQC + obj_mem_BNQC

        prob_with_bg_NHW = unpad(prob_with_bg_NHW, pad_P)
        return prob_with_bg_NHW

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_config.yaml')
def eval_vos(cfg: DictConfig):
    log.info(f'All configureations: {cfg}')
    if cfg['output_dir'] is not None:
        run_dir = cfg['output_dir']
    else:
        run_dir = HydraConfig.get().run.dir

    weights_path = cfg.weights

    # determine where to save the masks
    mask_output_root = path.join(run_dir, 'Annotations')
    score_output_root = path.join(run_dir, 'Scores')
    visualize_output_root = path.join(run_dir, 'Visualizations')

    # Setup dataset.
    data_cfg = get_dataset_cfg(cfg)
    image_dir = data_cfg.image_directory
    mask_dir = data_cfg.mask_directory
    size = data_cfg.size
    use_all_masks = data_cfg.get('use_all_masks')
    skip_frames = data_cfg.get('skip_frames')
    json_dir = data_cfg.get('json_directory')
    size_dir = data_cfg.get('size_directory')
    subset = data_cfg.get('subset')

    network = LIVOS(model_type='base').cuda().eval()
    network.load_weights(torch.load(weights_path, weights_only=True))

    dataset_name = cfg.dataset
    is_burst = ('burst' in dataset_name)
    if is_burst:
        dataset = BURSTTestDataset(
            image_dir=image_dir,
            json_dir=json_dir,
            size=size,
            skip_frames=skip_frames)
        burst_handler = BURSTResultHandler(dataset.json)        
    else:
        dataset = VOSTestDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            subset=subset,
            size_dir=size_dir,
            size=size,
            use_all_masks=use_all_masks,
            req_frames_json=json_dir)
    
    dataloader = dataset.get_datasets()
    total_process_time = 0
    total_frames = 0

    pbar = tqdm(dataloader, total=len(dataset))
    for video_reader in pbar:
        loader = DataLoader(
            video_reader, batch_size=None, shuffle=False, num_workers=4)
        video_name = video_reader.vid_name
        pbar.set_description(video_name)
        video_length = len(loader)

        try:
            processor = InferenceCore(network)
            saver = ResultSaver(
                mask_output_root,
                video_name,
                dataset=dataset_name,
                object_manager=processor.object_manager,
                use_long_id=video_reader.use_long_id,
                palette=video_reader.get_palette(),
                init_json=video_reader.sequence_json if is_burst else None)

            first_mask_loaded = False
            for i, data in enumerate(loader):
                with torch.amp.autocast('cuda', enabled=False):
                    image = data['rgb'].cuda()
                    mask = data.get('mask')
                    if mask is not None:
                        mask = mask.cuda()
                        first_mask_loaded = True
                
                    if not first_mask_loaded:
                        continue

                    objects = data.get('valid_labels')
                    if objects is not None:
                        objects = objects.tolist()

                    info = data['info']
                    frame_name = info['frame']
                    shape = info['shape']
                    resize_needed = info['resize_needed']
                    path_to_image = info['path_to_image']

                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    is_last_frame = (i == video_length - 1)
                    # torch.cuda.empty_cache()
                    prob = processor.step(image, mask, objects, is_last_frame)

                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1

                    if data_cfg.save_all or info['save']:
                        saver.process(
                            prob,
                            frame_name,
                            resize_needed=resize_needed,
                            shape=shape,
                            last_frame=is_last_frame,
                            path_to_image=path_to_image)
            saver.end()  
            if is_burst:
                burst_handler.add_sequence(saver.video_json)          

        except Exception as e:
            log.error(f'Runtime error at {video_name}')
            log.error(e)
            saver.end()
            raise e

    log.info(f'Total processing time: {total_process_time}')
    log.info(f'Total processed frames: {total_frames}')
    log.info(f'FPS: {total_frames / total_process_time}')
    log.info(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (1024**2)}')
    log.info(f'Max reserved memory (MB): {torch.cuda.max_memory_reserved() / (1024**2)}')

    make_zip(dataset_name, run_dir, cfg.exp_id, mask_output_root)
    if is_burst:
        burst_handler.dump(run_dir)


if __name__ == '__main__':
    eval_vos()