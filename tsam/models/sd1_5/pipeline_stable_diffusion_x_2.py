# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import torch.nn as nn
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torch.nn import functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.utils.torch_utils import randn_tensor
from pytorch_metric_learning import distances, losses


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        # this one is what's called in normal generation
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionPipelineX_2(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        
        ##_x
        self.attn_fetch_x = None
        ##_x

    def get_text_sa(self, prompt, device, num_images_per_prompt=1, negative_prompt=None, lora_scale=None):
        self.eos_idx = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            True,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=None,
        )

        return self.attn_fetch_x.store_text_sa(text_encoder = self.text_encoder)
        

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            eos_idx = torch.where(text_input_ids==49407)[1][0].item()
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds, eos_idx

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    @staticmethod
    def _compute_self_attn_loss_cos(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        latent_opt_config,
        self_attn_score_pairs=None,
        avg_text_sa_norm = None,

    ) -> torch.Tensor:
        """Computes the cosine similarity loss using the self attention of text encoder and cross attention maps."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if latent_opt_config.softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if latent_opt_config.softmax_normalize:
                attention_for_text_t_plus_one *= 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )
        
        cos = nn.CosineSimilarity(dim=0) # row-wise cosine similarity
        cos_loss = 0
        text_token_len = avg_text_sa_norm.shape[0]
        # make cross attn cos sim matrix
        cos_matrix = torch.eye(text_token_len, dtype=attention_for_text.dtype).to(attention_for_text.device)
        for row_idx in range(text_token_len):    
            for column_idx in range(row_idx+1):
                if row_idx == column_idx:
                    continue
                embedding_1 = attention_for_text[:, :, row_idx]
                embedding_2 = attention_for_text[:, :, column_idx]
                if latent_opt_config.do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=latent_opt_config.smoothing_kernel_size, sigma=latent_opt_config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                    
                    smoothing = GaussianSmoothing(
                        kernel_size=latent_opt_config.smoothing_kernel_size, sigma=latent_opt_config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_2 = smoothing(input).squeeze(0).squeeze(0)
                embedding_1 = embedding_1.view(-1)
                embedding_2 = embedding_2.view(-1)
                if latent_opt_config.softmax_normalize_attention_maps:
                    embedding_1 *= 100
                    embedding_1 = torch.nn.functional.softmax(embedding_1)
                    embedding_2 *= 100
                    embedding_2 = torch.nn.functional.softmax(embedding_2)
                    
                embedding_1 = embedding_1.to(attention_for_text.device)
                embedding_2 = embedding_2.to(attention_for_text.device)
                
                cos_score = cos(embedding_1, embedding_2)
                cos_matrix[row_idx, column_idx] = cos_score
        # breakpoint()
        cos_loss_lst = []
        for row_idx in range(1,text_token_len):

            cos_loss += (0.2*row_idx)*(1-cos(avg_text_sa_norm[row_idx, :row_idx], cos_matrix[row_idx, :row_idx]))

        return cos_loss

    @staticmethod
    def _compute_self_attn_loss_cos_attn_like(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        latent_opt_config,
        self_attn_score_pairs=None,
        avg_text_sa_norm = None,
    ) -> torch.Tensor:
        """Computes the cosine similarity loss using the self attention of text encoder and cross attention maps."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if latent_opt_config.softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if latent_opt_config.softmax_normalize:
                attention_for_text_t_plus_one *= 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )
        
        cos = nn.CosineSimilarity(dim=0)
        cos_loss = 0
        text_token_len = avg_text_sa_norm.shape[0]
        # make cross attn cos sim matrix
        cos_matrix = torch.eye(text_token_len, dtype=attention_for_text.dtype).to(attention_for_text.device)
        for row_idx in range(text_token_len):    
            for column_idx in range(row_idx+1):
                if row_idx == column_idx:
                    continue
                embedding_1 = attention_for_text[:, :, row_idx]
                embedding_2 = attention_for_text[:, :, column_idx]
                if latent_opt_config.do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=latent_opt_config.smoothing_kernel_size, sigma=latent_opt_config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                    
                    smoothing = GaussianSmoothing(
                        kernel_size=latent_opt_config.smoothing_kernel_size, sigma=latent_opt_config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_2 = smoothing(input).squeeze(0).squeeze(0)
                embedding_1 = embedding_1.view(-1)
                embedding_2 = embedding_2.view(-1)
                if latent_opt_config.softmax_normalize_attention_maps:
                    embedding_1 *= 100
                    embedding_1 = torch.nn.functional.softmax(embedding_1)
                    embedding_2 *= 100
                    embedding_2 = torch.nn.functional.softmax(embedding_2)
                    
                embedding_1 = embedding_1.to(attention_for_text.device)
                embedding_2 = embedding_2.to(attention_for_text.device)
                
                if latent_opt_config.cos1_or_cos2 == 'cos2':
                    cos_score = cos(embedding_1, embedding_2)
                elif latent_opt_config.cos1_or_cos2 == 'cos1':
                    norm1_score = torch.inner(embedding_1, embedding_2) / (torch.norm(embedding_1, p=1) * torch.norm(embedding_2, p=1))
                    cos_score = norm1_score*len(embedding_1)
                cos_matrix[row_idx, column_idx] = cos_score
        # breakpoint()
        attn_like_mat = cos_matrix/torch.sum(cos_matrix, dim=1, keepdim=True)
        
        cos_loss_lst = []
        for row_idx in range(1,text_token_len):
            if latent_opt_config.loss_type == 'cos_loss':
                cos_loss += (latent_opt_config.row_weight*(row_idx+1)/text_token_len)*(1-cos(avg_text_sa_norm[row_idx, :row_idx+1], attn_like_mat[row_idx, :row_idx+1]))
            elif latent_opt_config.loss_type == 'abs_loss':
                cos_loss += (latent_opt_config.row_weight*(row_idx+1)/text_token_len)*torch.sum(torch.abs(avg_text_sa_norm[row_idx, :row_idx+1]-attn_like_mat[row_idx, :row_idx+1]))

        return cos_loss

    @staticmethod
    def _compute_self_attn_loss(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        self_attn_score_pairs=None,
    ) -> torch.Tensor:
        """Computes the attend-and-contrast loss using the maximum attention value for each token."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if softmax_normalize:
                attention_for_text_t_plus_one *= 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )
        
        cos = nn.CosineSimilarity(dim=0)
        mse_loss = 0
        mse_fn = nn.MSELoss()
        for idx_pairs, self_attn_score in self_attn_score_pairs.items():
            idx_1 = idx_pairs[0]
            idx_2 = idx_pairs[1]
            # breakpoint()
            embedding_1 = attention_for_text[:, :, idx_1 - 1]
            embedding_2 = attention_for_text[:, :, idx_2 - 1]
            if do_smoothing:
                smoothing = GaussianSmoothing(
                    kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                ).to(attention_for_text.device)
                input = F.pad(
                    embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                )
                embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                
                smoothing = GaussianSmoothing(
                    kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                ).to(attention_for_text.device)
                input = F.pad(
                    embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                )
                embedding_2 = smoothing(input).squeeze(0).squeeze(0)
            embedding_1 = embedding_1.view(-1)
            embedding_2 = embedding_2.view(-1)
            if softmax_normalize_attention_maps:
                embedding_1 *= 100
                embedding_1 = torch.nn.functional.softmax(embedding_1)
                embedding_2 *= 100
                embedding_2 = torch.nn.functional.softmax(embedding_2)
                
            embedding_1 = embedding_1.to(attention_for_text.device)
            embedding_2 = embedding_2.to(attention_for_text.device)
            
            cos_score = cos(embedding_1, embedding_2)
            mse_loss += mse_fn(cos_score, torch.tensor(self_attn_score, dtype=attention_maps.dtype).to(attention_for_text.device)) 
            
        return mse_loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents


    
    def _perform_iterative_refinement_step_with_attn(
        self,
        latents: torch.Tensor,
        loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        latent_opt_config,
        attention_maps_t_plus_one: Optional[torch.Tensor] = None,
        self_attn_score_pairs=None,
        avg_text_sa_norm=None,
        eos_idx=None,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        cnt = 0 
        loss = 100.0
        # manually set
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.03*28} # 0.03 is average cosine distance
        loss_threshold = {5:0.14*4, 6:0.14*5, 7:0.14*6, 8:0.14*7, 9:0.14*8, 10:0.14*9} # 0.03 is average cosine distance # I added all settings for 8+
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.18*6}
        for cnt in range(30):
            # print("cnt", cnt)
            if cnt == 20 and loss > loss_threshold[eos_idx-1]:
                print("perform_iterative_refinement_step_with_attn break from (cnt == 20 and loss > loss_threshold[eos_idx-1])!")
                break
            
        # while loss > loss_threshold[eos_idx-1] and cnt < 30:
        #     cnt += 1
        #     print(cnt)
            
                
            # for iteration in range(refinement_steps):
            #     iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
            attention_maps = self.attn_fetch_x.storage

            if latent_opt_config.attn_like_loss:
                loss = self._compute_self_attn_loss_cos_attn_like(
                    attention_maps=attention_maps,
                    attention_maps_t_plus_one=attention_maps_t_plus_one,
                    latent_opt_config = latent_opt_config,
                    self_attn_score_pairs=self_attn_score_pairs,
                    avg_text_sa_norm=avg_text_sa_norm,
                )
            else:
                loss = self._compute_self_attn_loss_cos(
                    attention_maps=attention_maps,
                    attention_maps_t_plus_one=attention_maps_t_plus_one,
                    latent_opt_config=latent_opt_config,
                    self_attn_score_pairs=self_attn_score_pairs,
                    avg_text_sa_norm=avg_text_sa_norm,
                )

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)
        
        if cnt >= 30 - 1 and loss > loss_threshold[eos_idx-1]:
            print("refinement step ended with cnt > 30 and loss > threshold")
        
        if cnt >= 30 - 1 and loss < loss_threshold[eos_idx-1]:
            print("refinement step ended with cnt > 30 and loss < threshold")

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
        attention_maps = self.attn_fetch_x.storage

        if latent_opt_config.attn_like_loss:
            loss = self._compute_self_attn_loss_cos_attn_like(
                attention_maps=attention_maps,
                attention_maps_t_plus_one=attention_maps_t_plus_one,
                latent_opt_config = latent_opt_config,
                self_attn_score_pairs=self_attn_score_pairs,
                avg_text_sa_norm=avg_text_sa_norm,
            )
        else:
            loss = self._compute_self_attn_loss_cos(
                attention_maps=attention_maps,
                attention_maps_t_plus_one=attention_maps_t_plus_one,
                latent_opt_config = latent_opt_config,
                self_attn_score_pairs=self_attn_score_pairs,
                avg_text_sa_norm=avg_text_sa_norm,
            )
            
        return loss, latents
    


            
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_iter_to_alter: int = 25,
        steps_to_save_attention_maps = None,
        latent_opt_config = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # print a bunch of the info for debugging
        print("callback:", callback)
        print("callback_steps:", callback_steps)
        print("guidance_scale:", guidance_scale)
        print("self.unet.config.time_cond_proj_dim:", self.unet.config.time_cond_proj_dim)
        print("guidance_rescale:", guidance_rescale)
        print("clip_skip:", clip_skip)
        print("cross_attention_kwargs:", cross_attention_kwargs)
        print("self.do_classifier_free_guidance:", self.do_classifier_free_guidance)
        print("output_type", output_type)
        print("num_images_per_prompt", num_images_per_prompt)
        print("negative_prompt", negative_prompt)
        print("ip adapter stuff:", ip_adapter_image_embeds, ip_adapter_image)
        print("negative_prompt_embeds", negative_prompt_embeds)
        print("prompt_embeds", prompt_embeds)
        print("sigmas", sigmas)
        print("timesteps", timesteps)

        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds, eos_idx = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        if self.attn_fetch_x is not None:
            all_text_sa = self.attn_fetch_x.store_text_sa(text_encoder = self.text_encoder)
            if  max_iter_to_alter != 0:
                avg_block_text_sa = torch.mean(torch.stack(list(all_text_sa.values())), dim=0) 
        else:
            print('no attention fetch. attention maps are not being stored.')   


        if  max_iter_to_alter != 0:

            avg_block_text_sa_except_bos_eos = avg_block_text_sa[1:eos_idx,1:eos_idx]**latent_opt_config.k
            avg_block_text_sa_norm = avg_block_text_sa_except_bos_eos/(torch.sum(avg_block_text_sa_except_bos_eos, dim=1).unsqueeze(1)+1e-7) # renormalization
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        print("extra_step_kwargs", extra_step_kwargs)

        raise Exception("break point")
        

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            
            
            
            
            
            
            
            
            
        ## latent update
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = latent_opt_config.scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :]
            if self.do_classifier_free_guidance
            else prompt_embeds
        )

                
        attention_map = [{} for i in range(num_images_per_prompt)]
        attention_map_t_plus_one = None
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)
                    updated_latents = []
                    for j, (latent, text_embedding) in enumerate(
                        zip(latents, text_embeddings)
                    ):
                        # Forward pass of denoising with text conditioning
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)

                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()
                        # Get max activation value for each subject token
                        # attn_map = self._aggregate_attention()
                        
                        if self.attn_fetch_x is not None:
                            self.attn_fetch_x.store_attn_by_timestep(t.item(),self.unet)
                            attn_map = self.attn_fetch_x.storage
                            
                            ## 
                            # self.attn_fetch_x.store_key_by_timestep(t.item(),self.unet)
                            # key = self.attn_fetch_x.storage_key
                            # breakpoint()
                        else:
                            print('no attention_fetch. attention maps are not being stored.')
                    
                        # breakpoint()
                        if (steps_to_save_attention_maps and i in steps_to_save_attention_maps):
                            attention_map[j][i] = attn_map.detach().cpu()
                        
                        if max_iter_to_alter != 0:
                            if latent_opt_config.attn_like_loss:
                                loss = self._compute_self_attn_loss_cos_attn_like(
                                attention_maps=attn_map,
                                attention_maps_t_plus_one=attention_map_t_plus_one,
                                latent_opt_config=latent_opt_config,
                                # self_attn_score_pairs=self_attn_score_pairs,
                                avg_text_sa_norm = avg_block_text_sa_norm,
                            )
                            else:
                                loss = self._compute_self_attn_loss_cos(
                                    attention_maps=attn_map,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    latent_opt_config=latent_opt_config,
                                    # self_attn_score_pairs=self_attn_score_pairs,
                                    avg_text_sa_norm = avg_block_text_sa_norm,
                                )
                        # print("attn loss", loss)

                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in latent_opt_config.iterative_refinement_steps:

                                loss, latent = self._perform_iterative_refinement_step_with_attn(
                                    latents=latent,
                                    loss=loss,
                                    text_embeddings=text_embedding,
                                    step_size=step_size[i],
                                    t=t,
                                    latent_opt_config=latent_opt_config,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    # self_attn_score_pairs=self_attn_score_pairs,
                                    avg_text_sa_norm = avg_block_text_sa_norm,
                                    eos_idx=eos_idx,

                                )
                        # Perform gradient update
                        # breakpoint()
                        if i < max_iter_to_alter:
                            if loss != 0:
                                # print("update latent")
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=loss,
                                    step_size=step_size[i],
                                )
                        ##########
                        
                        
                        
                        
                        
                        
                        
                        
                        updated_latents.append(latent)

                    latents = torch.cat(updated_latents, dim=0)

                if latent_opt_config.add_previous_attention_maps and (latent_opt_config.previous_attention_map_anchor_step is None or i == latent_opt_config.previous_attention_map_anchor_step):

                    attention_map_t_plus_one = self.attn_fetch_x.storage

                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)
        
        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), attention_map
        return image, attention_map
    


    """
    call from run.py:
    pipe(
        prompt=prompt,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        max_iter_to_alter=max_iter_to_alter,
        steps_to_save_attention_maps=steps_to_save_attention_maps,
        latent_opt_config = latent_opt_config
    )  
    """
    @torch.no_grad()
    def reduced_call(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_iter_to_alter: int = 25,
        steps_to_save_attention_maps = None,
        latent_opt_config = None,
    ):
        r"""
        The call function to the pipeline for generation.

        NOTE: Iterative refinement is still called for steps past max_iter_to_alter

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.


        Returns:
            image, attention maps
        """

        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        """
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        """

        # required defaults:
        num_images_per_prompt = 1
        self._guidance_scale = 7.5


        # do_classifier_free_guidance is a self property that will always be true. When this is done, the negative prompt is set to ""


        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        

        device = self._execution_device

        # 3. Encode input prompt
        """
        In normal running, this will encode the normal prompt and provide an empty negative prompt for classifier free guidance
        """
        prompt_embeds, negative_prompt_embeds, eos_idx = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt=None, # removed         Passing in None for negative_prompt and negative_prompt_embeds results in the empty negative prompt: ""
            prompt_embeds=None, #removed
            negative_prompt_embeds=None, #removed
            lora_scale=None, #removed
            clip_skip=None, #removed
        )


        # Get text average text self attention matrix (including all tokens)
        """
        self.attn_fetch_x should be set from utils.py from:

            model.attn_fetch_x = AttnFetchSDX_3()

        This code pretty much creates a dictionary of the text encoder attention matrices and takes an average of all

        It doesn't seem to save anything onto the attn_fetch_x object

        """
        if self.attn_fetch_x is not None:
            all_text_sa = self.attn_fetch_x.store_text_sa(text_encoder = self.text_encoder)
            if  max_iter_to_alter != 0:
                avg_block_text_sa = torch.mean(torch.stack(list(all_text_sa.values())), dim=0) 
        else:
            print('no attention fetch. attention maps are not being stored.')



        """
        This section does the slicing off for the end token index

        It them does row normalization (rows sum to 1)
        """
        if  max_iter_to_alter != 0:

            avg_block_text_sa_except_bos_eos = avg_block_text_sa[1:eos_idx,1:eos_idx]**latent_opt_config.k
            avg_block_text_sa_norm = avg_block_text_sa_except_bos_eos/(torch.sum(avg_block_text_sa_except_bos_eos, dim=1).unsqueeze(1)+1e-7) # renormalization

        

        # concatenate  prompts since classifier free guidance is always true
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                                                            # normal prompt embed
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device #removed timesteps and sigmas arguments
        )
        # timesteps is now an array of scheduler indices (1-1000)

        
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        print("num latent channels:", num_channels_latents)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float16,#prompt_embeds.dtype,
            device,
            generator,
            #latents,
        )   
            
            
            
            
            
        ## latent update
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = latent_opt_config.scale_factor * np.sqrt(scale_range) # step size is used as step size for iterative refinement step

        print("step sizes", np.round(step_size, 2))


        # this grabs the prompt text embedding (not the negative)
        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :] #should just be 1:
            if self.do_classifier_free_guidance
            else prompt_embeds
        )

        
        
        attention_map = [{} for i in range(num_images_per_prompt)]
        attention_map_t_plus_one = None


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        print("num_warmup_steps:", num_warmup_steps)


        print("latents shape:", latents.shape)
        print("steps_to_save_attention_map:", steps_to_save_attention_maps)
        print("latent_opt_config.attn_like_loss:", latent_opt_config.attn_like_loss)
        print("max_iter_to_alter:", max_iter_to_alter)
        print("iterative_refinement_steps:", latent_opt_config.iterative_refinement_steps)

        print("device:", device)


        print("off-loading VAE and text_encoder to CPU during latent optimization")
        self.text_encoder.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()


        # self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            
            # run through all timesteps (scheduler indices)
            for i, t in enumerate(timesteps):

                if self.interrupt:
                    continue
                
                # This section runs the latent through the UNet to calculate the loss and update accordingly
                # This DOES NOT do anything for the actual denoising
                # 
                # The latent is updated at each step with an extra interative refinement called at iterative_refinement_steps
                #
                with torch.enable_grad():

                    # latents is the latent tensor (batch_size x 4 x 64x64)
                    latents = latents.clone().detach().requires_grad_(True) # this essentially copies the tensor and resets the gradients

                    # list for collecting the updated latents (will be set as latents at end)
                    updated_latents = []

                    # go through each latent and text embedding from each prompt
                    for j, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):

                        # Forward pass of denoising with text conditioning

                        # this adds a batch dimension to each (still batchx4x64x64)
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)



                        # do a forward pass on the U-Net so attention maps are computed
                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            #cross_attention_kwargs=None,
                        ).sample

                        # set all UNet gradients to 0
                        self.unet.zero_grad()


                        # Get max activation value for each subject token
                        # attn_map = self._aggregate_attention()
                        
                        if self.attn_fetch_x is not None:

                            # this saves the U-Net attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
                            # this saves it to self.attn_fetch_x.storage then reads it back
                            self.attn_fetch_x.store_attn_by_timestep(t.item(),self.unet)
                            attn_map = self.attn_fetch_x.storage
                            
                            ## 
                            # self.attn_fetch_x.store_key_by_timestep(t.item(),self.unet)
                            # key = self.attn_fetch_x.storage_key
                            # breakpoint()
                        else:
                            print('no attention_fetch. attention maps are not being stored.')
                    

                        # breakpoint()
                        if (steps_to_save_attention_maps and i in steps_to_save_attention_maps): # with default setup, this is all stetps
                            attention_map[j][i] = attn_map.detach().cpu()
                        

                        if max_iter_to_alter != 0:
                            if latent_opt_config.attn_like_loss: # will be false by default
                                # deleted
                                pass
                            else:
                                # this calculates the attention loss with the most recently added attention map
                                loss = self._compute_self_attn_loss_cos(
                                    attention_maps=attn_map,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    latent_opt_config=latent_opt_config,
                                    # self_attn_score_pairs=self_attn_score_pairs,
                                    avg_text_sa_norm = avg_block_text_sa_norm,
                                )
                        print("attn loss:", loss.item())


                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in latent_opt_config.iterative_refinement_steps:

                            print("Starting iterative refinement for step:", i)

                            loss, latent = self._perform_iterative_refinement_step_with_attn(
                                latents=latent,
                                loss=loss,
                                text_embeddings=text_embedding,
                                step_size=step_size[i],
                                t=t,
                                latent_opt_config=latent_opt_config,
                                attention_maps_t_plus_one=attention_map_t_plus_one, # attention map saved from previous denoising NOT THE NEXT ONE
                                # self_attn_score_pairs=self_attn_score_pairs,
                                avg_text_sa_norm = avg_block_text_sa_norm,
                                eos_idx=eos_idx,
                            )
                        
                        
                        # Perform gradient update
                        # breakpoint()
                        if i < max_iter_to_alter:
                            if loss != 0:
                                print(f"updated latent (step{i})")
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=loss,
                                    step_size=step_size[i],
                                )
                        ##########
                        
                        
                        
                        updated_latents.append(latent)
                    

                    latents = torch.cat(updated_latents, dim=0)

                
                # the latent has now been updated, so everything else if for the actual denoising step


                # this condition is true in normal runs
                if latent_opt_config.add_previous_attention_maps and (latent_opt_config.previous_attention_map_anchor_step is None or i == latent_opt_config.previous_attention_map_anchor_step):

                    attention_map_t_plus_one = self.attn_fetch_x.storage

                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                ).sample
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # removed **extra_step_kwargs which is normally {}


                # call the callback, if provided (removed)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        print("Loading back VAE and text_encoder for final steps")
        self.text_encoder.to(device)
        self.vae.to(device)

        output_type = "pil"
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), attention_map
        return image, attention_map
    




    """
    A version of the generation function that has unused features removed and implements our bias loss
    """
    @torch.no_grad()
    def bias_reduced_call(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_iter_to_alter: int = 25,
        steps_to_save_attention_maps = None,
        latent_opt_config = None,

        bias_prompt_form = None,
        bias_info = None,          # list of dictionaries
        occupation_info = None     # just a dictionary
    ):
        r"""
        The call function to the pipeline for simple bias loss generation.

        NOTE: Iterative refinement is still called for steps past max_iter_to_alter

        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            max_iter_to_alter (`int`, *optional*, defaults to 50):
                Maximum iteration for implementing single step TSAM loss calls 
            steps_to_save_attention_maps (`List[int]`, *optional*):
                List of iterations to save the UNet attention map into the attention_map dictionary, which is returned at end
            bias_prompt_form ('str`, *required*):
                Bias prompt form to insert bias words into. Ex: "A photo of a {}"
                The resulting prompt embeddings are then sliced to insert the optimized bias word embedding from OVAM
            bias_info (`List[dict]`, *required*)
                list of dictionaries with bias information (should be two)
                At a minimum, these should include:
                    'word' : bias_word (ex: male/man/female/woman)
                    'opt_embedding' : numpy (768,) array of optimized bias word embedding from OVAM
            occupation_info (`dict`, *required*)
                dictionary with information for the occupation
                At a minimum, should include:
                    'word' : occupation_word (ex: nurse/teacher/doctor)
                The index of this word will be found later using the tokenizer tokens


        Returns:
            image, attention maps
        """

        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        """
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        """

        # required defaults:
        num_images_per_prompt = 1
        self._guidance_scale = 7.5


        # do_classifier_free_guidance is a self property that will always be true. When this is done, the negative prompt is set to ""


        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        

        device = self._execution_device

        # 3. Encode input prompt
        """
        In normal running, this will encode the normal prompt and provide an empty negative prompt for classifier free guidance
        """
        prompt_embeds, negative_prompt_embeds, eos_idx = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt=None, # removed         Passing in None for negative_prompt and negative_prompt_embeds results in the empty negative prompt: ""
            prompt_embeds=None, #removed
            negative_prompt_embeds=None, #removed
            lora_scale=None, #removed
            clip_skip=None, #removed
        )


        # Get text average text self attention matrix (including all tokens)
        """
        self.attn_fetch_x should be set from utils.py from:

            model.attn_fetch_x = AttnFetchSDX_3()

        This code pretty much creates a dictionary of the text encoder attention matrices and takes an average of all

        It doesn't seem to save anything onto the attn_fetch_x object

        """
        if self.attn_fetch_x is not None:
            # I'm zeroing everything out for safety
            for i in range(12):
                self.text_encoder.text_model.encoder.layers[i].self_attn.dummy = 0
            
            all_text_sa = self.get_text_sa(prompt=prompt, device=self.device)
            
            if  max_iter_to_alter != 0:
                avg_block_text_sa = torch.mean(torch.stack(list(all_text_sa.values())), dim=0)
            
            avg_block_text_sa = avg_block_text_sa.clone()

        else:
            print('no attention fetch. attention maps are not being stored.')


        """
        This section does the slicing off for the end token index

        It then does row normalization (rows sum to 1)
        """
        if  max_iter_to_alter != 0:

            avg_block_text_sa_except_bos_eos = avg_block_text_sa[1:eos_idx,1:eos_idx]**latent_opt_config.k
            avg_block_text_sa_norm = avg_block_text_sa_except_bos_eos/(torch.sum(avg_block_text_sa_except_bos_eos, dim=1).unsqueeze(1)+1e-7) # renormalization


        # concatenate  prompts since classifier free guidance is always true
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                                                            # normal prompt embed



        
        # Prepare bias loss function information
        
        # get occupation idx (add 'idx' entry to dictionary)
        occ_token_id = self.tokenizer(occupation_info['word'], add_special_tokens=False).input_ids[0]
        prompt_text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        occupation_index_matches = torch.where(prompt_text_inputs['input_ids'] == occ_token_id)

        if len(occupation_index_matches) == 0:
            raise Exception("Occupation word does not appear in main prompt")
        else:
            occupation_info['idx'] = occupation_index_matches[1][0].item()


        # for each bias dictionary, add an entry for prompt_embedding, and idx
        for bias_dict in bias_info:

            # Form bias prompt string by injecting the word into the bias prompt form
            bias_prompt = bias_prompt_form.format(bias_dict['word'])

            # get list of tokenizer ids
            bias_text_inputs = self.tokenizer(bias_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # find bias word index in the prompt token ids
            b_text_input_ids = bias_text_inputs.input_ids
            bias_token_id = self.tokenizer(bias_dict['word'], add_special_tokens=False).input_ids[0]
            bias_index = torch.where(b_text_input_ids == bias_token_id)[1][0].item()


            # create bias prompt encoding with pipe's text_encoder
            b_text_input_ids = b_text_input_ids.to(self.text_encoder.device)
            bias_prompt_embeds = self.text_encoder(b_text_input_ids, attention_mask=None)[0] # 0 index grabs last hidden states

            
            # insert the OVAM optimized bias word embedding into the bias prompt embedding
            bias_prompt_embeds = bias_prompt_embeds.clone()
            bias_prompt_embeds[0, bias_index, :] = torch.tensor(bias_dict['opt_embedding']).to(prompt_embeds.dtype)
            

            # add to dictionary
            bias_dict['idx'] = bias_index
            bias_dict['prompt_embedding'] = bias_prompt_embeds


            # NOTE: I don't think this makes a difference from just encoding the bias word and then inserting the optimized embedding

        
        



                                            
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device #removed timesteps and sigmas arguments
        )
        # timesteps is now an array of scheduler indices (1-1000)

        
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            #latents,
        )   
            
            
            
            
            
        ## latent update
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = latent_opt_config.scale_factor * np.sqrt(scale_range) # step size is used as step size for iterative refinement step

        #print("step sizes", np.round(step_size, 2))


        # this grabs the prompt text embedding (not the negative)
        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :] #should just be 1:
            if self.do_classifier_free_guidance
            else prompt_embeds
        )

        
        
        attention_map = [{} for i in range(num_images_per_prompt)]
        attention_map_t_plus_one = None









        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        print("num_warmup_steps:", num_warmup_steps)


        #print("latents shape:", latents.shape)
        #print("steps_to_save_attention_map:", steps_to_save_attention_maps)
        #print("latent_opt_config.attn_like_loss:", latent_opt_config.attn_like_loss)
        #print("max_iter_to_alter:", max_iter_to_alter)
        #print("iterative_refinement_steps:", latent_opt_config.iterative_refinement_steps)


        # self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            
            # run through all timesteps (scheduler indices)
            for i, t in enumerate(timesteps):

                if self.interrupt:
                    continue
                
                # This section runs the latent through the UNet to calculate the loss and update accordingly
                # This DOES NOT do anything for the actual denoising
                # 
                # The latent is updated at each step with an extra interative refinement called at iterative_refinement_steps
                #
                with torch.enable_grad():

                    # latents is the latent tensor (batch_size x 4 x 64x64)
                    latents = latents.clone().detach().requires_grad_(True) # this essentially copies the tensor and resets the gradients

                    # list for collecting the updated latents (will be set as latents at end)
                    updated_latents = []

                    # go through each latent and text embedding from each prompt
                    for j, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):

                        # Forward pass of denoising with text conditioning

                        # this adds a batch dimension to each (now 1xbatchx4x64x64)
                        latent = latent.unsqueeze(0)

                        text_embedding = text_embedding.unsqueeze(0) # [1, 77, 768]



                        # do a forward pass on the U-Net so cross attention maps are computed
                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            #cross_attention_kwargs=None,
                        ).sample

                        # set all UNet gradients to 0
                        self.unet.zero_grad()


                        # Get max activation value for each subject token
                        # attn_map = self._aggregate_attention()
                        
                        if self.attn_fetch_x is not None:

                            # this saves the U-Net attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
                            # this saves it to self.attn_fetch_x.storage then reads it back
                            self.attn_fetch_x.store_attn_by_timestep(t.item(),self.unet)
                            attn_map = self.attn_fetch_x.storage
                            
                            ## 
                            # self.attn_fetch_x.store_key_by_timestep(t.item(),self.unet)
                            # key = self.attn_fetch_x.storage_key
                            # breakpoint()
                        else:
                            print('no attention_fetch. attention maps are not being stored.')
                    

                        # breakpoint()
                        if (steps_to_save_attention_maps and i in steps_to_save_attention_maps): # with default setup, this is all stetps
                            attention_map[j][i] = attn_map.detach().cpu()
                        

                        if max_iter_to_alter != 0:
                            if latent_opt_config.attn_like_loss: # will be false by default
                                # deleted
                                pass
                            else:
                                # this calculates the attention loss with the most recently added attention map
                                loss = self._compute_self_attn_loss_cos(
                                    attention_maps=attn_map,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    latent_opt_config=latent_opt_config,
                                    # self_attn_score_pairs=self_attn_score_pairs,
                                    avg_text_sa_norm = avg_block_text_sa_norm,
                                )
                                print(f"Iter {i}, T-SAM attn loss:", loss.item())


                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in latent_opt_config.iterative_refinement_steps:

                            print("Starting T-SAM iterative refinement for step:", i)

                            loss, latent = self._perform_iterative_refinement_step_with_attn(
                                latents=latent,
                                loss=loss,
                                text_embeddings=text_embedding,
                                step_size=step_size[i],
                                t=t,
                                latent_opt_config=latent_opt_config,
                                attention_maps_t_plus_one=attention_map_t_plus_one, # attention map saved from previous denoising NOT THE NEXT ONE
                                # self_attn_score_pairs=self_attn_score_pairs,
                                avg_text_sa_norm = avg_block_text_sa_norm,
                                eos_idx=eos_idx,
                            )
                            print(f"T-SAM loss after iter refinement: {loss.item()}")
                        


                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in latent_opt_config.bias_refinement_steps:

                            print("Starting bias iterative refinement for step:", i)

                            loss, latent = self._bias_iterative_refinement_step(
                                latents=latent,
                                loss=loss,
                                text_embeddings=text_embedding,
                                step_size=step_size[i],
                                t=t,
                                latent_opt_config=latent_opt_config,
                                attention_maps_t_plus_one=attention_map_t_plus_one, # attention map saved from previous denoising NOT THE NEXT ONE
                                # self_attn_score_pairs=self_attn_score_pairs,
                                avg_text_sa_norm = avg_block_text_sa_norm,
                                eos_idx=eos_idx,
                                
                                bias_info=bias_info,             # list of dictionaries
                                occupation_info=occupation_info  # just a dictionary  
                            )
                            print(f"T-SAM loss after BIAS iter refinement: {loss.item()}")

                            
                        
                        
                        # Perform gradient update
                        # breakpoint()
                        if i < max_iter_to_alter:
                            if loss != 0:
                                print(f"updated latent (step {i})")
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=loss,
                                    step_size=step_size[i],
                                )
                        ##########
                        
                        
                        
                        updated_latents.append(latent)
                    

                    latents = torch.cat(updated_latents, dim=0)

                
                # the latent has now been updated, so everything else if for the actual denoising step


                # this condition is true in normal runs
                if latent_opt_config.add_previous_attention_maps and (latent_opt_config.previous_attention_map_anchor_step is None or i == latent_opt_config.previous_attention_map_anchor_step):

                    attention_map_t_plus_one = self.attn_fetch_x.storage

                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                ).sample
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # removed **extra_step_kwargs which is normally {}


                # call the callback, if provided (removed)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        
        output_type = "pil"
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), attention_map
        return image, attention_map



    """
    Implements iterative loss on bias setup.

    Can be set with max_cnt=1 in latent_opt_config to mimick single step losses

    """
    def _bias_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        latent_opt_config,
        attention_maps_t_plus_one: Optional[torch.Tensor] = None,
        self_attn_score_pairs=None,
        avg_text_sa_norm=None,
        eos_idx=None,

        bias_info=None,        # list of dictionaries
        occupation_info=None   # just a dictionary
    ):
        """
        Performs modified latent refinment based on matching the cross attention matrices of the two biased prompt embeddings

        After the iterative bias loss has been completed, it performs one T-SAM loss before returning (that's what most of the imports are for)

        bias_info (`List[dict]`, *required*)
                list of dictionaries with bias information (should be two)
                these should include:
                    'word' : bias_word (ex: male/man/female/woman)
                    'opt_embedding' : numpy (768,) array of optimized bias word embedding from OVAM
                    'prompt_embedding' : tensor (768x77) prompt emedding for biased prompt to run through UNet (OVAM optimized embedding already inserted)
                    'idx' : int index of bias word in prompt embedding
        occupation_info (`dict`, *required*)
            dictionary with information for the occupation
            Should include:
                'word' : occupation_word (ex: nurse/teacher/doctor)
                'idx'  : index of occupation_word in text_embeddings (main prompt)
        """

        # create details as alias for latent_opt_config.bias_loss_function_details
        # determine max cnt (set by details)
        details = None
        max_cnt=30
        if latent_opt_config.bias_loss_function_details:
            details = latent_opt_config.bias_loss_function_details
            if 'max_cnt' in details:
                max_cnt=details['max_cnt']
        

        # set break threshold
        bias_loss_threshold = 0.01
        if latent_opt_config.bias_loss_threshold:
            bias_loss_threshold=latent_opt_config.bias_loss_threshold


        cnt = 0 
        loss = 100.0
        original_occ_maps = []
        # manually set
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.03*28} # 0.03 is average cosine distance
        # loss_threshold = {5:0.14*4, 6:0.14*5, 7:0.14*6, 8:0.14*7, 9:0.14*8, 10:0.14*9} # 0.03 is average cosine distance # I added all settings for 8+
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.18*6}
        for cnt in range(max_cnt):

            latents = latents.clone().detach().requires_grad_(True)

            # get the cross attention map for the occupation
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            # function to get cross attention map matrix based on biased loss settings (ex: what UNet layer)
            prompt_attention_maps = self.bias_loss_get_unet_cross_attention_map(t, latent_opt_config.bias_loss_function_details)

            all_occ_attn_maps = self.attn_fetch_x.get_unet_cross_attn_maps(self.unet) # get a list of all cross attention maps (for displaying)

            # get ca matrix for occupation
            occ_ca_maps = []
            for map in prompt_attention_maps:
                occ_ca_maps.append(map[:, :, occupation_info['idx']].squeeze())
            # occ_ca_map = prompt_attention_maps[:, :, occupation_info['idx']].squeeze()
                if(cnt==0):
                    original_occ_maps.append(occ_ca_maps[-1].detach())
            # print('prompt',prompt_attention_maps.shape,occ_ca_map.shape)


            # get the cross attention map for first bias
            self.unet(latents, t, encoder_hidden_states=bias_info[0]['prompt_embedding']).sample
            self.unet.zero_grad()

            # function to get cross attention map matrix based on biased loss settings (ex: what UNet layer)
            b1_attention_maps = self.bias_loss_get_unet_cross_attention_map(t, latent_opt_config.bias_loss_function_details)

            all_b1_attn_maps = self.attn_fetch_x.get_unet_cross_attn_maps(self.unet) # get a list of all cross attention maps (for displaying)
            b1_ca_maps = []
            # get ca matrix for first bias
            for map in b1_attention_maps:
                b1_ca_maps.append(map[:, :, bias_info[0]['idx']].squeeze())
            # b1_ca_map = b1_attention_maps[:, :, bias_info[0]['idx']].squeeze()




            # get the cross attention map for second bias
            self.unet(latents, t, encoder_hidden_states=bias_info[1]['prompt_embedding']).sample
            self.unet.zero_grad()

            # function to get cross attention map matrix based on biased loss settings (ex: what UNet layer)
            b2_attention_maps = self.bias_loss_get_unet_cross_attention_map(t, latent_opt_config.bias_loss_function_details)

            all_b2_attn_maps = self.attn_fetch_x.get_unet_cross_attn_maps(self.unet) # get a list of all cross attention maps (for displaying)

            # get ca matrix for second bias
            b2_ca_maps = []
            for map in b2_attention_maps:
                b2_ca_maps.append(map[:, :, bias_info[1]['idx']].squeeze())
            # b2_ca_map = b2_attention_maps[:, :, bias_info[1]['idx']].squeeze()
            
            # print('b1',b1_attention_maps.shape,b1_ca_map.shape)
            # print('b2',b2_attention_maps.shape,b2_ca_map.shape)
            
            # make matplot lib plot s
            # plot functions at end of file
            if latent_opt_config.show_first_bias_map and cnt == 0:

                separate_scales = details.get("separate_scales") if details else None # get key-value if details exists
                
                if details and 'display' in details:
                    if details['display'] == 'all':
                        plot_all_bias_maps(occ_maps=all_occ_attn_maps,
                                           b1_maps=all_b1_attn_maps,
                                           b2_maps=all_b2_attn_maps,
                                           occ_info=occupation_info,
                                           bias_info=bias_info,
                                           separate_scales=separate_scales)
                    else:
                        plot_bias_maps(occ_map = occ_ca_maps[0].detach(),
                                       b1_map = b1_ca_maps[0].detach(), 
                                       b2_map = b2_ca_maps[0].detach(),
                                       occ_info=occupation_info,
                                       bias_info=bias_info,
                                       separate_scales=separate_scales)                    
                else:
                    plot_bias_maps(occ_map = occ_ca_maps[0].detach(),
                                   b1_map = b1_ca_maps[0].detach(), 
                                   b2_map = b2_ca_maps[0].detach(),
                                   occ_info=occupation_info,
                                   bias_info=bias_info,
                                   separate_scales=separate_scales)



            # TODO: implement loss function
            loss=0
            for occ_ca_map, b1_ca_map, b2_ca_map in zip(original_occ_maps, b1_ca_maps, b2_ca_maps):
                loss += self._compute_bias_self_attn_loss_cos(
                    occ_attention_map=occ_ca_map,
                    b1_attention_map=b1_ca_map,
                    b2_attention_map=b2_ca_map,
                    ideal_ratio=1.2
                )

            """
            loss = self.some_loss_function(
                occ_attention_map=occ_ca_map,
                b1_attention_map=b1_ca_map,
                b2_attention_map=b2_ca_map,
            )
            """
            # all of these maps should be a 2D matrix

            # import warnings
            # warnings.warn("Bias loss function has not been implemented yet. Setting to 0 by default")
            # loss = 0

            print(f'bias loss iter {cnt}: {loss.item()}')




            if loss < bias_loss_threshold:
                print(f"bias loss breaking from loss < {bias_loss_threshold}")
                break

            if loss != 0:
                latents = self._update_latent(latents, loss, 10*step_size)
        


        if latent_opt_config.show_last_bias_map:

            print("\n\nShowing last bias map:")

            separate_scales = details.get("separate_scales") if details else None # get key-value if details exists
            
            if details and 'display' in details:
                if details['display'] == 'all':
                    plot_all_bias_maps(occ_maps=all_occ_attn_maps,
                                        b1_maps=all_b1_attn_maps,
                                        b2_maps=all_b2_attn_maps,
                                        occ_info=occupation_info,
                                        bias_info=bias_info,
                                        separate_scales=separate_scales)
                else:
                    plot_bias_maps(occ_map = occ_ca_map.detach(),
                                    b1_map = b1_ca_map.detach(), 
                                    b2_map = b2_ca_map.detach(),
                                    occ_info=occupation_info,
                                    bias_info=bias_info,
                                    separate_scales=separate_scales)                    
            else:
                plot_bias_maps(occ_map = occ_ca_map.detach(),
                                b1_map = b1_ca_map.detach(), 
                                b2_map = b2_ca_map.detach(),
                                occ_info=occupation_info,
                                bias_info=bias_info,
                                separate_scales=separate_scales)
        
        # do T-SAM loss before returning. Added if statement to be shrinkable in VSCode
        if True:
            # Run one more time but don't compute gradients and update the latents.
            # We just need to compute the new loss - the grad update will occur below
            latents = latents.clone().detach().requires_grad_(True)
            _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
            attention_maps = self.attn_fetch_x.storage

            if latent_opt_config.attn_like_loss:
                loss = self._compute_self_attn_loss_cos_attn_like(
                    attention_maps=attention_maps,
                    attention_maps_t_plus_one=attention_maps_t_plus_one,
                    latent_opt_config = latent_opt_config,
                    self_attn_score_pairs=self_attn_score_pairs,
                    avg_text_sa_norm=avg_text_sa_norm,
                )
            else:
                loss = self._compute_self_attn_loss_cos(
                    attention_maps=attention_maps,
                    attention_maps_t_plus_one=attention_maps_t_plus_one,
                    latent_opt_config = latent_opt_config,
                    self_attn_score_pairs=self_attn_score_pairs,
                    avg_text_sa_norm=avg_text_sa_norm,
                )
            
        return loss, latents


    @staticmethod
    def _compute_bias_self_attn_loss_cos(
        occ_attention_map=torch.Tensor,
        b1_attention_map=torch.Tensor,
        b2_attention_map=torch.Tensor,
        ideal_ratio=2 #this is desired ration of cosine similarity for b1 to b2
    ):
        cos = nn.CosineSimilarity(dim=0)
        
        occ_embed = occ_attention_map.detach().view(-1)
        b1_embed = b1_attention_map.view(-1)
        b2_embed = b2_attention_map.view(-1)

        b1_occ_cos_score = cos(b1_embed,occ_embed)
        b2_occ_cos_score = cos(b2_embed,occ_embed)
        
        # print(f'b1: {b1_occ_cos_score.item():.3f}, {b1_embed.mean():.3f}, b2: {b2_occ_cos_score.item():.3f}, {b2_embed.mean():.3f}')

        scale_factor = 0.5
        b_mean = (b1_embed+b2_embed)/2
        b2_sharp = b2_embed/b_mean.mean()
        b1_sharp=b1_embed/b_mean.mean()
        print(f'b1: {b1_occ_cos_score.item():.3f}, {b1_sharp.mean():.3f}, {b1_embed.mean():.3f}, b2: {b2_occ_cos_score.item():.3f}, {b2_sharp.mean():.3f}, {b2_embed.mean():.3f}')
        b2_scaled = ideal_ratio*scale_factor*b2_sharp
        b1_scaled = scale_factor*b1_sharp/ideal_ratio
        mse_loss = torch.nn.functional.mse_loss(b2_scaled,b1_scaled)
        print(f'mse loss: {mse_loss}')
        loss = max(0.3,1-b1_occ_cos_score)*max(0.3,1-b2_occ_cos_score)*mse_loss
        # pred = torch.cat([b1_occ_cos_score.reshape(-1),b2_occ_cos_score.reshape(-1)])
        # target = torch.tensor([ideal_ratio,1],device=pred.device)
        # loss = 1 - cos(pred,target)
        return mse_loss

    """
    Function to retrieve the cross attention map from a specific layer in the UNet

    NOTE: if no specific layer is specified, then this returns the average of the two bottleneck layers (default TSAM)d

    To specify a layer, this expects the following entries in the latent_opt_config.bias_loss_function_details dictionary:

    - 'block_class' : str,       'CrossAttnDownBlock2D' or 'CrossAttnUpBlock2D'
    - 'total_map_size' : int,    16*16, 32*32, 64*64

    NOTE: Down blocks have 2 cross attention layers each while Up blocks have three

    Returned attention maps are averaged over the layers of the block
        
    """
    def bias_loss_get_unet_cross_attention_map(self, t, bias_loss_settings):
        details = bias_loss_settings

        attention_maps = []

        if details:
            if 'block_class' in details and details['block_class'] is not None:
                if 'total_map_size' in details and details['total_map_size'] is not None:
                    for size in details['total_map_size']:
                        attention_maps.append(
                            self.attn_fetch_x.get_specific_unet_cross_attn_map(self.unet, 
                                                                                block_class = details['block_class'],
                                                                                total_map_size = size) # (s,s,77)
                        )
                    # attention_maps = self.attn_fetch_x.get_specific_unet_cross_attn_map(self.unet, 
                                                                                        # block_class = details['block_class'],
                                                                                        # total_map_size = details['total_map_size']) # (s,s,77)
                else:
                    raise Exception("for bias_loss_function_details: block_class is defined but total_map_size isn't")
            else:
                self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
                attention_maps = self.attn_fetch_x.storage # (16,16,77)
        else:
            self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
            attention_maps = self.attn_fetch_x.storage # (16,16,77)
        
        return attention_maps


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
    

















def plot_all_bias_maps(occ_maps, b1_maps, b2_maps, occ_info, bias_info, separate_scales):

    import matplotlib.pyplot as plt
    
    num_maps = len(occ_maps)

    # run through each map level
    for i in range(num_maps):

        # grab cross attention maps
        occ_np = occ_maps[i].detach().cpu()[:, :, occ_info['idx']].squeeze().numpy()
        b1_np  =  b1_maps[i].detach().cpu()[:, :, bias_info[0]['idx']].squeeze().numpy()
        b2_np  =  b2_maps[i].detach().cpu()[:, :, bias_info[1]['idx']].squeeze().numpy()

        print("ca map shape:", occ_np.shape)
        
        # Shared scale
        vmin = min(occ_np.min(), min(b1_np.min(), b2_np.min()))
        vmax = max(occ_np.max(), max(b1_np.max(), b2_np.max()))

        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        for ax, data, title in zip(axes, [occ_np, b1_np, b2_np], [occ_info['word'], bias_info[0]['word'], bias_info[1]['word']]):
            # separate scales
            if separate_scales:
                im = ax.imshow(data)
                fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.10)

            # same scale (colorbar added later)
            else:
                im = ax.imshow(data, vmin=vmin, vmax=vmax)
            
            ax.set_title(title)

            """
            # Add numbers to each square
            if occ_np.shape[0] == 16:
                for (i, j), val in np.ndenumerate(data):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
            """

        # Shared colorbar
        if not separate_scales:
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()





def plot_bias_maps(occ_map, b1_map, b2_map, occ_info, bias_info, separate_scales):
    import matplotlib.pyplot as plt

    # Move to CPU + convert to numpy for matplotlib
    occ_np = occ_map.detach().cpu().numpy()
    b1_np  =  b1_map.detach().cpu().numpy()
    b2_np  =  b2_map.detach().cpu().numpy()

    # Shared scale
    vmin = min(occ_np.min(), min(b1_np.min(), b2_np.min()))
    vmax = max(occ_np.max(), max(b1_np.max(), b2_np.max()))

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    for ax, data, title in zip(axes, [occ_np, b1_np, b2_np], [occ_info['word'], bias_info[0]['word'], bias_info[1]['word']]):
        # separate scales
        if separate_scales:
            im = ax.imshow(data)
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.10)

        # same scale (colorbar added later)
        else:
            im = ax.imshow(data, vmin=vmin, vmax=vmax)
        
        ax.set_title(title)

        """
        # Add numbers to each square
        if occ_np.shape[0] == 16:
            for (i, j), val in np.ndenumerate(data):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
        """

    # Shared colorbar
    if not separate_scales:
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

