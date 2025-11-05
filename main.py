# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:00:10 2025

@author: JDawg
"""

from diffusers import DiffusionPipeline
from generate_images import gen_imgs
from sd_evaluation_pipeline import clip_eval
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

generation_prompt = 'A nurse'
style = ', photorealistic'
images = gen_imgs(prompt = generation_prompt + style, samples= 20) #this will eventually be multiple images per sample


#one attribute at a time (e.g. man/woman or white/black/hispanic,etc.)
evaluation_dict = {
                   'gender': ['man', 'woman'],
                   'race'  : ['white', 'black', 'asian', 'hispanic'],
                   'weight': ['fat', 'skinny'],
                   'age'   : ['child', 'teenager', 'adult', 'elderly person']
                   }
for key in evaluation_dict.keys():
    probs = clip_eval(images, evaluation_dict[key], plot = True)
    df = pd.DataFrame(data = probs.detach().cpu().numpy(), columns = evaluation_dict[key])
    evaluation_dict[key] = df




