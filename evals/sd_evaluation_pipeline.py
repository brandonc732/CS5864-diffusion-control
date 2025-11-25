# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:19 2025

@author: JDawg
"""
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
import numpy as np 

def clip_eval(images, attributes = ['man', 'woman'], plot = False):
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")    
    inputs = processor(text=attributes, images=images, return_tensors="pt", padding=True, do_convert_rgb=False)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1) 
    
    if plot: 
        'Plot the generated images with their associated attribute probability'
        for idx in range(len(images)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

            # show original image
            axes[0].imshow(images[idx])
            axes[0].axis('off')

            # show probabilities as horizontal bar
            axes[1].barh(range(len(attributes)), probs[idx].detach().numpy(), tick_label=attributes)
            axes[1].set_xlim(0, 1.0)

            plt.tight_layout()
            plt.show()
    
    return probs


