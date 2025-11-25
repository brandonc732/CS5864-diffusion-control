# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
import evals
# import distance
# import embedding
# import io_util
import numpy as np
from datasets import load_dataset
import os
from PIL import Image
import os
import tempfile
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)



def create_temp_image_dir(image_list):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="temp_eval_images_")

    for i, arr in enumerate(image_list):
        # Ensure uint8 format
        if not np.issubdtype(arr.dtype, np.uint8):
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB if needed
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        
        img = Image.fromarray(arr)
        img.save(os.path.join(temp_dir, f"{i:05d}.png"))

    return temp_dir


#ONLY NEED THIS FUNCTION CALL 
def compute_cmmd(eval_dir = './eval_dir', ref_dir = './ref_dir', ref_embed_file=None, batch_size=32, max_count=-1, background_samples = 750):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    #downloads the reference dataset (sample of 750 from 70k), should be about .5 GBs
    if not os.path.isdir(ref_dir):
        print(f'downloading flickr-faces-hq-dataset from hugging face at: {ref_dir}')
        dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)
        os.makedirs(ref_dir, exist_ok=True)
        i = 0
        for sample in tqdm(dataset):
            img = sample['image']
            img.save(os.path.join(ref_dir,f"{i:05d}.png"))
            i+=1
            if i > background_samples:
                break
        print("Saved", i, "images to", ref_dir)

    #create a directory of images if the directory doesn't exist
    if isinstance(eval_dir, list):
        eval_dir = create_temp_image_dir(eval_dir)

    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = evals.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = evals.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = evals.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = evals.mmd(ref_embs, eval_embs)

    if 'temp_eval_images_' in eval_dir:
        print(f'deleting temporary directory: {eval_dir}')
        shutil.rmtree(eval_dir)

    
    return val.numpy()


def main(argv):
    import os
    dir1 = os.path.join(os.getcwd(), r"generated_images")
    dir2 = os.path.join(os.getcwd(), r"reference_images")
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
    )


if __name__ == "__main__":
    app.run(main)


