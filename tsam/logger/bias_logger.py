import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union, Any

import numpy as np
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt

TensorLike = Union[torch.Tensor, np.ndarray]

class BiasLogger:
    """
    Logger for bias-related data during diffusion model training or inference.

    Collects:
      - cross-attention per UNet layer
      - bias-loss scalars
      - cosine similarities for bias concepts
      - images or latents (auto-decodes via VAE)
      - metadata

    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        vae: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        decode_fn: Optional[Callable[[TensorLike], Image.Image]] = None,
        compress_npz: bool = True,
        image_format: str = "PNG",
        latent_scale: float = 0.18215,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.vae = vae
        self.image_processor = image_processor
        if decode_fn is not None:
            self.decode_fn = decode_fn
        elif vae is not None:
            self.decode_fn = self._default_decode_fn
        else:
            self.decode_fn = None

        self.compress_npz = bool(compress_npz)
        self.image_format = image_format.upper()
        self.latent_scale = latent_scale

        # core storage
        self._attn: Dict[str, List[torch.Tensor]] = {}
        self._bias_losses: List[float] = []
        self._cosine_sims: List[float] = []            # NEW: track cosine similarity for bias concepts
        self._images: List[Image.Image] = []
        self.metadata: Dict[str, object] = {}

        self.attention_plots = []

    # internal method
    def _default_decode_fn(self, latents: torch.Tensor) -> Image.Image:
        
        with torch.no_grad():
            img = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        """
        latents = latents / self.latent_scale
        latents = latents.to(self.vae.dtype)
        with torch.no_grad():
             = self.vae.decode(latents).sample  # B,3,H,W
        """
        """
        img = img[0] if img.ndim == 4 else img  # handle batch=1
        img = (img.clamp(-1, 1) + 1) / 2
        img = (img * 255).byte().cpu().permute(1, 2, 0).numpy()
        """

        img = self.image_processor.postprocess(img, output_type='pil', do_denormalize=[True])

        return img[0]

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x
        raise TypeError("Expected tensor or numpy array")

    # logging
    def log_attention(self, layer_name: str, attn: TensorLike) -> None:
        t = self._to_tensor(attn).detach().cpu()
        self._attn.setdefault(layer_name, []).append(t)
    
    def log_attention_plot(self, attn_plot : Image) -> None:
        self.attention_plots.append(attn_plot)

    def log_bias_loss(self, loss: Union[float, torch.Tensor]) -> None:
        v = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss)
        self._bias_losses.append(v)

    def log_cosine_similarity(self, sim: Union[float, torch.Tensor]) -> None:
        """Call this to log cosine similarity (e.g. between concept vectors)."""
        v = float(sim.detach().cpu().item()) if isinstance(sim, torch.Tensor) else float(sim)
        self._cosine_sims.append(v)

    def log_image(self, img_or_latent: Union[Image.Image, TensorLike]) -> None:
        if isinstance(img_or_latent, Image.Image):
            self._images.append(img_or_latent.copy())
            return

        if self.decode_fn is None:
            raise ValueError("No decode_fn or VAE provided; cannot decode latent.")

        t = self._to_tensor(img_or_latent).detach()#.cpu()
        if t.ndim == 3 and t.shape[0] == 4:  # latent (4,H,W)
            pil = self.decode_fn(t)
        elif t.ndim == 4 and t.shape[1] == 4:  # batch latent
            pil = self.decode_fn(t)
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
        self._cosine_sims.clear()
        self._images.clear()
        self.metadata.clear()

    # saving methods (kept the same with small change for cosine)
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
            except Exception:
                pass

    def save_bias_losses(self):
        with open(self.save_dir / "bias_losses.txt", "w") as f:
            for v in self._bias_losses:
                f.write(f"{v}\n")

        with open(self.save_dir / "bias_losses.json", "w") as f:
            json.dump({"bias_losses": self._bias_losses, "cosine_sims": self._cosine_sims}, f, indent=2)

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

    #animation methods
    def _fig_to_pil(self, fig) -> Image.Image:
        """Convert a matplotlib figure to PIL Image and close the figure."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        buf.close()
        plt.close(fig)
        return im

    def animate_attention(self, layer_name: str, out_fname: Optional[Union[str, Path]] = None, fps: int = 4, normalize: bool = True):
        """
        Create an animated GIF for attention maps of a given layer over time.
        - layer_name: must match a logged layer via log_attention
        - out_fname: path to output GIF (if None, saved under save_dir/attention_maps/<layer>/animation.gif)
        - fps: frames per second (controls 'duration')
        - normalize: if True, normalize entire sequence to the same min/max for consistent colors
        """
        if layer_name not in self._attn:
            raise ValueError(f"No attention stored for '{layer_name}'")

        tensors = [t.numpy() for t in self._attn[layer_name]]
        # tensors shape may vary; try to collapse to 2D attention maps per step
        frames = []
        # find global vmin/vmax if needed
        if normalize:
            global_min = min(t.min() for t in tensors)
            global_max = max(t.max() for t in tensors)
        else:
            global_min, global_max = None, None

        for idx, arr in enumerate(tensors):
            # attempt to convert to 2D heatmap:
            # Common attention shapes: (heads, q_len, k_len) or (q_len, k_len) or (batch, heads, q, k)
            a = arr
            if a.ndim == 4:  # batch, heads, q, k -> collapse heads and batch
                a = a.reshape(-1, a.shape[-2], a.shape[-1]).mean(axis=0)
            elif a.ndim == 3:  # heads, q, k -> mean over heads
                a = a.mean(axis=0)
            elif a.ndim == 2:
                pass
            else:
                # fallback: try flattening to square-ish 2D
                flat = a.ravel()
                L = int(np.sqrt(flat.size))
                a = flat[: L * L].reshape(L, L)

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            im = ax.imshow(a, aspect='auto', vmin=global_min, vmax=global_max)
            ax.set_title(f"{layer_name} — step {idx}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            pil = self._fig_to_pil(fig)
            frames.append(pil.convert("RGBA"))

        # write GIF
        if out_fname is None:
            out_fname = self._attn_dir(layer_name) / "attention_animation.gif"
        else:
            out_fname = Path(out_fname)
            out_fname.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / max(1, fps))
        frames[0].save(out_fname, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2)
        return out_fname

    def animate_images(self, out_fname: Optional[Union[str, Path]] = None, fps: int = 4):
        """
        Create GIF from logged decoded images (self._images).
        Use this to show how the decoded image evolves each latent iteration.
        """
        if not self._images:
            raise ValueError("No images logged to animate.")

        frames = [img.convert("RGBA") for img in self._images]
        if out_fname is None:
            out_fname = self.save_dir / "images" / "image_progression.gif"
            (self.save_dir / "images").mkdir(parents=True, exist_ok=True)
        else:
            out_fname = Path(out_fname)
            out_fname.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / max(1, fps))
        frames[0].save(out_fname, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2)
        return out_fname
    
    def animate_ca_plots(self, out_fname: Optional[Union[str, Path]] = None, fps: int = 4):
        """
        Create GIF from logged cross attention plots (self.attention_pots).
        Use this to show how the cross attention evolves each latent iteration.
        """
        if not self.attention_plots:
            raise ValueError("No images logged to animate.")

        frames = [img.convert("RGBA") for img in self.attention_plots]
        if out_fname is None:
            out_fname = self.save_dir / "images" / "image_progression.gif"
            (self.save_dir / "images").mkdir(parents=True, exist_ok=True)
        else:
            out_fname = Path(out_fname)
            out_fname.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / max(1, fps))
        frames[0].save(out_fname, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2)
        return out_fname

    def animate_bias_metrics(self, out_fname: Optional[Union[str, Path]] = None, fps: int = 4, window: Optional[int] = None):
        """
        Animate the progression of bias-loss and cosine similarity over steps.)
        """
        steps = max(len(self._bias_losses), len(self._cosine_sims))
        if steps == 0:
            raise ValueError("No bias metrics logged.")

        bias = list(self._bias_losses)
        cos = list(self._cosine_sims)
        # pad shorter lists
        if len(bias) < steps:
            bias += [np.nan] * (steps - len(bias))
        if len(cos) < steps:
            cos += [np.nan] * (steps - len(cos))

        frames = []
        x = np.arange(steps)

        for i in range(steps):
            fig = plt.figure(figsize=(6, 3.5))
            ax = fig.add_subplot(111)
            lo = 0 if window is None else max(0, i - window + 1)
            xs = x[lo:i+1]
            ys_bias = np.array(bias[lo:i+1], dtype=float)
            ys_cos = np.array(cos[lo:i+1], dtype=float)

            ax.plot(xs, ys_bias, label="bias_loss")
            ax.plot(xs, ys_cos, label="cosine_sim")
            ax.scatter([i], [bias[i]] if not np.isnan(bias[i]) else [np.nan], s=50)
            ax.set_xlim(left=lo, right=max(lo + 1, i + 1))
            ax.set_xlabel("step")
            ax.set_title("Bias metrics over time")
            ax.legend(loc="upper right")
            ax.grid(True)
            pil = self._fig_to_pil(fig)
            frames.append(pil.convert("RGBA"))

        if out_fname is None:
            out_fname = self.save_dir / "bias_metrics_progression.gif"
        else:
            out_fname = Path(out_fname)
            out_fname.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / max(1, fps))
        frames[0].save(out_fname, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2)
        return out_fname









    def _write_mp4(self, frames, out_fname: Path, fps: int = 4, codec: str = "libx264"):
        """
        Write MP4 using imageio (ffmpeg backend). Frames should be PIL Images.
        """
        import numpy as np
        import imageio.v2 as imageio  # pip install imageio imageio-ffmpeg

        out_fname = Path(out_fname)
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        # Ensure consistent size + even dims for broad MP4 compatibility (yuv420p)
        w, h = frames[0].size
        if (w % 2) == 1: w += 1
        if (h % 2) == 1: h += 1

        writer = imageio.get_writer(
            str(out_fname),
            fps=fps,
            codec=codec,
            pixelformat="yuv420p",
        )

        try:
            for im in frames:
                if im.size != (w, h):
                    im = im.resize((w, h))
                arr = np.asarray(im.convert("RGB"))  # MP4 doesn't support alpha
                writer.append_data(arr)
        finally:
            writer.close()

        return out_fname