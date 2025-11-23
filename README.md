# CS5864-diffusion-control
Repository for CS5864 final project about controlling stable diffusion generation bias by manipulating the U-Net model's attention mechanisms


# Method overview

Our general method will involve:
- itendifying race or gender bias within the CLIP text encoder
- Using OVAM to identify image generation bias with these bias terms
- Using a T-SAM-like loss to enforce a set generation bias guidance.


# Repository overview

The `ovam` and `tsam` folders contain our modified versions of the original OVAM and T-SAM github repositories.

All installation files and instructions are found in the `_environment_info` folder. 

Below are overviews of what changes are made or need to be made to T-SAM and OVAM 


## T-SAM folder

**Brandon's current changes:**

- For simplification, I've created two reduced_call functions that remove any unused code chunks and provide much better comments on what's going on
- For investigation, I added parts within the code to display cross attention matrix development at specified steps
- For testing, I created some initial bias guidance frameworks, but they are largely incorrect and need to be overhauled.
- For testing, I created a jupyter lab notebook to play around with the generation without having to save images and stuff

**Thing's to be done:**
- Identify text encoder self attention bias within prompts
  -  Consider layer combining methods other than simple averaging
  -  If possible, try to quantify a desired ratio between bias terms
- Add a more organized way to select what UNet layer to enforce bias loss at
  - Maybe create a separate class for handling T-SAM and bias loss
  - Consider a way to combine layers (OVAM does upsampling with interpolation)
- Create a more organized separate class that's used collecting, logging, and saving:
  - Cross attention matrices at UNet layers
  - bias loss values
  - decoded latents (image over generation)
- Bring back options (lost in reduced) for:
  - Gaussian smoothing
  - Different T-SAM loss measurements
- Do a direct intergration of OVAM
  - I just approximated by running different prompt through the U-Net
- Parameter tuning for generation process.


## OVAM folder
- Figure out how to make an environment that can run OVAM without changing the current T-SAM packages setup.
- Create a simple call to get refined text tokens for given bias terms







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



