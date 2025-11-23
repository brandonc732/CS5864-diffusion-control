# CS5864-diffusion-control
Repository for CS5864 final project about controlling stable diffusion generation bias by manipulating the U-Net model's attention mechanisms
```python 
pip install transformers==4.55.4 diffusers==0.35.2 scipy ftfy accelerate "tokenizers>=0.21,<0.23"
```


# Source papers

**Text Self-Attention Maps (T-SAM)**

Focuses on aligning the attention matrices between the CLIP encoder and U-Net diffuser with cosine similarity.

The GitHub they provide shows how they construct their custom Stable Diffusion 1.5 pipeline with the ability to change its attention matrices at given diffusion steps. This will probably be our main source for laying out our model.

[github](https://github.com/t-sam-diffusion/code)
[arxiv](https://arxiv.org/pdf/2411.15236)



**Bias-Map**

Focuses on using cross attention maps from OVAM and the overlap between them for bias and occupation terms to quantity where inherent bias may lie in the generation. Their recent update to the paper employs a bayesian setup for using this to guide more fair generation 

(github not provided)
[arxiv](https://www.arxiv.org/pdf/2509.13496)



**OVAM**

At its base, OVAM shows how we can simply run a denoising step with a different prompt to show how words not in the original prompt attend to the current latent. Their primary contribution is a way to optimize the individual text tokens to get more accuracy cross attention matrices.

[github](https://github.com/vpulab/ovam)
[arxiv](https://arxiv.org/pdf/2403.14291)



