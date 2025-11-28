
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

TensorLike = Union[torch.Tensor, np.ndarray]


class BiasLogger:
    """
    logger that:
      - collects cross-attention tensors per named UNet layer
      - collects bias-loss scalars per step
      - collects decoded images per step (accepts PIL.Image OR latent tensors + decode_fn)
      - saves artifacts to disk in organized layout

    Args:
      save_dir: directory to write artifacts
      decode_fn: optional callable to convert latent -> PIL.Image (if you will pass latents)
      compress_npz: whether to use compressed npz for attention arrays
      image_format: image format for saving (PNG by default)
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        decode_fn: Optional[Callable[[TensorLike], Image.Image]] = None,
        compress_npz: bool = True,
        image_format: str = "PNG",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.decode_fn = decode_fn
        self.compress_npz = bool(compress_npz)
        self.image_format = image_format.upper()

        # in-memory stores
        self._attn: Dict[str, List[torch.Tensor]] = {}
        self._bias_losses: List[float] = []
        self._images: List[Image.Image] = []
        self.metadata: Dict[str, object] = {}
    # logging methods
    def log_attention(self, layer_name: str, attn: TensorLike) -> None:
        """Log attention for a named UNet layer. Accepts torch.Tensor or numpy array."""
        if isinstance(attn, np.ndarray):
            t = torch.from_numpy(attn).cpu()
        elif isinstance(attn, torch.Tensor):
            t = attn.detach().cpu()
        else:
            raise TypeError("attn must be a torch.Tensor or numpy.ndarray")
        self._attn.setdefault(layer_name, []).append(t)

    def log_bias_loss(self, loss: Union[float, torch.Tensor]) -> None:
        """Append a scalar bias-loss (float or 0-dim torch tensor)."""
        if isinstance(loss, torch.Tensor):
            val = float(loss.detach().cpu().item())
        else:
            val = float(loss)
        self._bias_losses.append(val)

    def log_image(self, img_or_latent: Union[Image.Image, TensorLike]) -> None:
        """
        Log an image for this step. Accepts:
          - PIL.Image.Image (stored directly)
          - latent tensor / numpy array -> decoded via decode_fn (must be provided)
        """
        if isinstance(img_or_latent, Image.Image):
            self._images.append(img_or_latent.copy())
            return

        if self.decode_fn is None:
            raise ValueError("No decode_fn provided; cannot decode latent to image.")
        latent = img_or_latent
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu()
        pil = self.decode_fn(latent)
        if not isinstance(pil, Image.Image):
            raise TypeError("decode_fn must return a PIL.Image")
        self._images.append(pil.copy())

    def set_metadata(self, **kwargs) -> None:
        self.metadata.update(kwargs)

    def clear(self) -> None:
        """Clear in-memory buffers (does not delete on-disk files)."""
        self._attn.clear()
        self._bias_losses.clear()
        self._images.clear()
        self.metadata.clear()

    #saving implementation 

    def _attn_layer_dir(self, layer_name: str) -> Path:
        d = self.save_dir / "attention_maps" / layer_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_attention_maps(self) -> None:
        """Save per-step attention files and try to save stacked archive when possible."""
        for layer_name, tensors in self._attn.items():
            layer_dir = self._attn_layer_dir(layer_name)
            for idx, t in enumerate(tensors):
                arr = t.numpy()
                fname = f"step_{idx:03d}"
                if self.compress_npz:
                    np.savez_compressed(layer_dir / f"{fname}.npz", arr=arr)
                else:
                    np.save(layer_dir / f"{fname}.npy", arr)
            # try stacked archive if shapes match
            try:
                stacked = np.stack([t.numpy() for t in tensors], axis=0)
                all_name = f"{layer_name}_all_steps"
                if self.compress_npz:
                    np.savez_compressed(layer_dir / f"{all_name}.npz", stacked=stacked)
                else:
                    np.save(layer_dir / f"{all_name}.npy", stacked)
            except Exception:
                pass

    def save_bias_losses(self) -> None:
        txt_path = self.save_dir / "bias_losses.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for v in self._bias_losses:
                f.write(f"{v}\n")
        json_path = self.save_dir / "bias_losses.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"bias_losses": self._bias_losses}, f, indent=2)

    def save_images(self) -> None:
        img_dir = self.save_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(self._images):
            out = img_dir / f"step_{i:03d}.{self.image_format.lower()}"
            img.save(out, format=self.image_format)

    def save_metadata(self) -> None:
        if not self.metadata:
            return
        with open(self.save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def finalize(self, clear_after_save: bool = False) -> None:
        """Write everything collected to disk."""
        self.save_attention_maps()
        self.save_bias_losses()
        self.save_images()
        self.save_metadata()
        if clear_after_save:
            self.clear()

    # Utilities & context manager
    def num_steps(self) -> int:
        if self._images:
            return len(self._images)
        if self._attn:
            return max(len(v) for v in self._attn.values())
        return len(self._bias_losses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.finalize(clear_after_save=False)
        except Exception:
            pass
 