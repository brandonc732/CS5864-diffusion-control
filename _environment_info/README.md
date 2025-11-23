# Overview

This folder includes all module installation instructions and optional environment files to install.


# Installing by modules:

Jordan's comments for Stable diffusion 1.5 with diffusers
```python 
pip install transformers==4.55.4 diffusers==0.35.2 scipy ftfy accelerate "tokenizers>=0.21,<0.23"
```

<br>
<br>

**WARNING:** The diffusers github recommends the following command to install:
```bash
pip install --upgrade diffusers[torch]
```
HOWEVER, if using anaconda, this will install pytorch a second time in pip, which will cause a torch import to crash python. So I only recommend you run this command without the [torch]



# Installing by yml file

If you're using anaconda, you can install directly from the yml files within this folder using the following command:

```bash
conda env create -f filename.yml
```


