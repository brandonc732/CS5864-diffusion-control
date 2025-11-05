# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:58:50 2025

@author: JDawg
"""
from diffusers import DiffusionPipeline
import torch
import numpy as np


#generates images from stable diffusion... will need to resample from same noise when doing attention mods here...
def gen_imgs(prompt = '', samples = 10):
    
    pipe_15 = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe_15 = pipe_15.to("cuda")
    image_list = [np.asarray(pipe_15(prompt).images[0], dtype = int) for _ in range(samples)]
    
    return image_list