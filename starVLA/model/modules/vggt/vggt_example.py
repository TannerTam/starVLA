# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from vggt.models.vggt import VGGT

import torchvision.transforms as T
from PIL import Image


from accelerate.logging import get_logger

logger = get_logger(__name__)

import torch.nn as nn


class _VGGT_Interface(nn.Module):

    def __init__(self):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        model = VGGT.from_pretrained("facebook/VGGT-1B")
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

        for sample_images in batch_images_pil:
            processed_imgs = []
            for img in sample_images:
                # Check if we should use self.vggt_transform or the local resize.
                # Here we use the local resize_transform to match get_vggt_emds logic,
                # but ensure it is converted to tensor if the transform doesn't do it.
                # Assuming T.Resize returns PIL, we usually need ToTensor() or valid conversion.
                # If self.vggt_transform includes ToTensor, we should use that, 
                # but standard T.Resize keeps it as PIL unless composed.
                
                # Standard torchvision pipeline usually requires ToTensor for the model
                # I will add ToTensor() to ensure it interacts with stack() correctly.
                
                img_resized = resize_transform(img)
                img_tensor = T.functional.to_tensor(img_resized) 
                
                # Note: If your model expects specific normalization (mean/std), 
                # ensure it is applied here. I am keeping it raw based on your snippet.
                processed_imgs.append(img_tensor)
            # Stack views -> [S, 3, 518, 518]
            sample_tensor = torch.stack(processed_imgs)
            batch_tensors.append(sample_tensor)

        vggt_input = torch.stack(batch_tensors)
        # 3. Model Inference (Logic from get_vggt_emds)
        # Using autocast context for mixed precision
        with torch.autocast("cuda", dtype=torch.float16):
            aggregated_tokens_list, _ = self.model.aggregator(
                vggt_input, # Passed the prepared 5D tensor
            )

        return aggregated_tokens_list[-1]


if __name__ == "__main__":
    vggt = _VGGT_Interface()
    pass
