# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from vggt.models.vggt import VGGT

import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image


from accelerate.logging import get_logger

logger = get_logger(__name__)

import torch.nn as nn


class _VGGT_Interface(nn.Module):

    def __init__(self, model_path: str = "./playground/Pretrained_models/VGGT-1B"):
        """
        Initialize the VGGT wrapper.
        Load VGGT from local path instead of HuggingFace Hub.
        """
        super().__init__()

        model = VGGT.from_pretrained(model_path)
        self.model = model


    def forward(
        self,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = self.model(
                **kwargs,
            )
        return predictions

    def aggregator(
        self,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.float16):
            aggregated_tokens_list, ps_idx = self.model.aggregator(
                **kwargs,
            )
        return aggregated_tokens_list, ps_idx
    
    def get_vggt_embeddings(self, batch_images_pil: List[List[Image.Image]], **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        resize_transform = T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC)
        batch_tensors = []
        max_views = max(len(sample_images) for sample_images in batch_images_pil)  # Find max number of views
        for sample_images in batch_images_pil:
            processed_imgs = []
            for img in sample_images:
                img_resized = resize_transform(img)
                img_tensor = T.functional.to_tensor(img_resized) 
                processed_imgs.append(img_tensor)
            
            # Pad if necessary to match max_views
            num_current_views = len(processed_imgs)
            if num_current_views < max_views:
                # Create zero padding tensors
                padding_tensor = torch.zeros_like(processed_imgs[0])
                for _ in range(max_views - num_current_views):
                    processed_imgs.append(padding_tensor)
            
            # Stack views -> [S, 3, 518, 518] where S = max_views
            sample_tensor = torch.stack(processed_imgs)
            batch_tensors.append(sample_tensor)
        # Now all tensors have the same shape [max_views, 3, 518, 518]
        vggt_input = torch.stack(batch_tensors)
        
        device = self.model.aggregator._resnet_mean.device
        vggt_input = vggt_input.to(device)
        
        # 3. Model Inference (Logic from get_vggt_emds)
        # Using autocast context for mixed precision
        with torch.autocast("cuda", dtype=torch.float16):
            aggregated_tokens_list, _ = self.model.aggregator(
                vggt_input, # Passed the prepared 5D tensor
            )
        features = aggregated_tokens_list[-1]
        B, S, P, Dim = features.shape
        features = features.reshape(B * S, P, Dim)
        
        # Apply pooling
        features_permuted = features.permute(0, 2, 1)  
        features_pooled = F.max_pool1d(features_permuted, kernel_size=10, stride=10)  
        features_final = features_pooled.permute(0, 2, 1)  
        
        features_final = features_final.reshape(B, S, -1, Dim)
        
        return features_final


if __name__ == "__main__":
    vggt = _VGGT_Interface()
    pass
