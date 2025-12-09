# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:00:10 2025

@author: JDawg
"""

from diffusers import DiffusionPipeline
from generate_images import gen_imgs
import evals
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from PIL import Image

generation_prompt = 'A nurse, male or female'

style = ', colored image, photorealistic'
images = gen_imgs(prompt = generation_prompt + style, samples= 2) #this will eventually be multiple images per sample


####EVALUATION STUFF###

# helper for loading the generated images.
def dir(folder):
    """Load all images from a directory into a numpy array."""
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            imgs.append(img)
    return imgs

# Runs clip fairness and cmmd realism eval 
# loads the given dir from the above helper function.
def evaluate(folder_path, attribute_dict, ref_dir="./"):

    print(f"\nfolder path: {folder_path}")

    images = dir(folder_path)
    if len(images) == 0:
        raise ValueError(f"No images found in {folder_path}")
    print(f"Loaded {len(images)} images.")

    # Fairness clip
    fairness = {}
    for attributes, labels in attribute_dict.items():
        probs = evals.clip_eval(images, labels, plot=False).detach().cpu().numpy()
        df = pd.DataFrame(probs, columns=labels)
        fairness[attributes] = df

    # realism retain/lost
    cmmd = evals.compute_cmmd(
        eval_dir=folder_path,
        ref_dir=ref_dir,
        background_samples=300
    )

    return fairness, cmmd

#CLIP attributes for probability of a ceratin image containing a specific 
#attribute. Simply softmax the logit outputs of CLIP.  

#one attribute is calculated at a time (e.g. man/woman or white/black/hispanic,etc.)
evaluation_dict = {
                   'gender': ['man', 'woman'],
                #    'race'  : ['white', 'black', 'asian', 'hispanic'],
                #    'weight': ['fat', 'skinny'],
                #    'age'   : ['child', 'teenager', 'adult', 'elderly person']
                   }
for key in evaluation_dict.keys():
    probs = evals.clip_eval(images, evaluation_dict[key], plot = True)
    df = pd.DataFrame(data = probs.detach().cpu().numpy(), columns = evaluation_dict[key])
    evaluation_dict[key] = df


#https://arxiv.org/html/2401.09603v1/
#CMMD essentially compares the distributions of two datasets through their CLIP embeddings.
#This requires an external dataset to compare our generated images against. In this case we use 
#a few samples from ffhq -- a high res dataset of faces. 

#The initial run will be slow...
#input can be either a list of arrays, or a directory of saved images (png/jpeg). 
cmmd_vals = evals.compute_cmmd(images, ref_dir = r"C:\Users\dogbl\Downloads\cmmd_backgroundss", background_samples = 3)
print(cmmd_vals)