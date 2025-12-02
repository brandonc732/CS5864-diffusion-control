import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union, Any

import numpy as np
import torch
from PIL import Image

TensorLike = Union[torch.Tensor, np.ndarray]


class BiasLogger:
    """
    Logger for bias-related data during diffusion model training or inference.

    Collects:
      - cross-attention per UNet layer
      - bias-loss scalars
      - images or latents (auto-decodes via VAE)
      - metadata
    
    Automatically handles:
      - per-step saving
      - latent decoding (if VAE or decode_fn is given)
      - batched inputs
      - consistent directory structure
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        vae: Optional[Any] = None,
        decode_fn: Optional[Callable[[TensorLike], Image.Image]] = None,
        compress_npz: bool = True,
        image_format: str = "PNG",
        latent_scale: float = 0.18215,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.vae = vae
        if decode_fn is not None:
            self.decode_fn = decode_fn
        elif vae is not None:
            self.decode_fn = self._default_decode_fn
        else:
            self.decode_fn = None

        self.compress_npz = bool(compress_npz)
        self.image_format = image_format.upper()
        self.latent_scale = latent_scale

        self._attn: Dict[str, List[torch.Tensor]] = {}
        self._bias_losses: List[float] = []
        self._images: List[Image.Image] = []
        self.metadata: Dict[str, object] = {}

    # internal method
    #default needed for VAE decoding and image saving.
    def _default_decode_fn(self, latents: torch.Tensor) -> Image.Image:
        latents = latents / self.latent_scale
        latents = latents.to(self.vae.dtype)
        with torch.no_grad():
            img = self.vae.decode(latents).sample  # B,3,H,W
        img = img[0] if img.ndim == 4 else img  # handle batch=1
        img = (img.clamp(-1, 1) + 1) / 2
        img = (img * 255).byte().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(img)

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x
        raise TypeError("Expected tensor or numpy array")

 #logging methods

    def log_attention(self, layer_name: str, attn: TensorLike) -> None:
        t = self._to_tensor(attn).detach().cpu()
        self._attn.setdefault(layer_name, []).append(t)

    def log_bias_loss(self, loss: Union[float, torch.Tensor]) -> None:
        v = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss)
        self._bias_losses.append(v)

    def log_image(self, img_or_latent: Union[Image.Image, TensorLike]) -> None:
        if isinstance(img_or_latent, Image.Image):
            self._images.append(img_or_latent.copy())
            return

        if self.decode_fn is None:
            raise ValueError("No decode_fn or VAE provided; cannot decode latent.")

        t = self._to_tensor(img_or_latent).detach().cpu()
        if t.ndim == 3 and t.shape[0] == 4:  # latent (4,H,W)
            pil = self.decode_fn(t)
        elif t.ndim == 4 and t.shape[1] == 4:  # batch latent
            pil = self.decode_fn(t[0])
        else:
            # assume image tensor
            if t.ndim == 3 and t.shape[0] in (1, 3):
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                img = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                pil = Image.fromarray(img)
            else:
                raise TypeError(f"Unsupported tensor shape {t.shape}")
        self._images.append(pil.copy())

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def clear(self):
        self._attn.clear()
        self._bias_losses.clear()
        self._images.clear()
        self.metadata.clear()

#saving methods

    def _attn_dir(self, layer: str):
        d = self.save_dir / "attention_maps" / layer
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_attention_maps(self):
        for layer, tensors in self._attn.items():
            d = self._attn_dir(layer)
            for i, t in enumerate(tensors):
                arr = t.numpy()
                fname = f"step_{i:03d}"
                path = d / f"{fname}.npz" if self.compress_npz else d / f"{fname}.npy"
                if self.compress_npz:
                    np.savez_compressed(path, arr=arr)
                else:
                    np.save(path, arr)

            # try stacked archive
            try:
                stacked = np.stack([t.numpy() for t in tensors], axis=0)
                allfile = d / f"{layer}_all_steps.npz"
                np.savez_compressed(allfile, stacked=stacked)
            except:
                pass

    def save_bias_losses(self):
        with open(self.save_dir / "bias_losses.txt", "w") as f:
            for v in self._bias_losses:
                f.write(f"{v}\n")

        with open(self.save_dir / "bias_losses.json", "w") as f:
            json.dump({"bias_losses": self._bias_losses}, f, indent=2)

    def save_images(self):
        img_dir = self.save_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(self._images):
            path = img_dir / f"step_{i:03d}.{self.image_format.lower()}"
            img.save(path, format=self.image_format)

    def save_metadata(self):
        if self.metadata:
            with open(self.save_dir / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)

    def finalize(self, clear_after_save=False):
        self.save_attention_maps()
        self.save_bias_losses()
        self.save_images()
        self.save_metadata()
        if clear_after_save:
            self.clear()

    def num_steps(self):
        if self._images:
            return len(self._images)
        if self._attn:
            return max(len(v) for v in self._attn.values())
        return len(self._bias_losses)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.finalize()
