# import torch
# from diffusers import AutoPipelineForText2Image

# # Load the model (this will download about 6GB on first run)
# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo", 
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
#     variant="fp16" if torch.cuda.is_available() else None
# )

# if torch.cuda.is_available():
#     pipe.to("cuda")

# def generate_broll_images(prompt, output_path):
#     # num_inference_steps=1 is what makes Turbo so fast
#     image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
#     image.save(output_path)
#     return output_path

import torch
import os
import gc
import streamlit as st
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

# Add this BEFORE you load the pipeline
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

@st.cache_resource
def load_pipeline(model_id=None):
    # Setup device and data types
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    # Accept a passed model_id, otherwise fallback to environment/default.
    model_id = model_id or os.environ.get("BROLL_MODEL_ID", "dreamlike-art/dreamlike-photoreal-2.0")

    flush()
    print(f"--- Loading image generation model {model_id} on {DEVICE.upper()} ---")

    if "sdxl" in model_id.lower():
        pipeline_class = StableDiffusionXLPipeline
        variant = "fp16" if DEVICE == "cuda" else None
    else:
        pipeline_class = StableDiffusionPipeline
        variant = None

    pipeline_kwargs = {"torch_dtype": DTYPE}
    if variant is not None:
        pipeline_kwargs["variant"] = variant

    pipe = pipeline_class.from_pretrained(
        model_id,
        **pipeline_kwargs,
    )

    pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    return pipe


# Setup device and data types
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# # 1. Clear memory before starting
# flush()
# print(f"--- Loading SDXL Turbo on {DEVICE.upper()} (Surgical Load) ---")

# # We load the specific class for SDXL, which skips the 'HunyuanDiT' import check
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/sdxl-turbo", 
#     torch_dtype=DTYPE, 
#     variant="fp16" if DEVICE == "cuda" else None
# )

# pipe.to(DEVICE)

# # Optional: Reduces memory usage on mid-range laptops
# if DEVICE == "cuda":
#     pipe.enable_attention_slicing()

def generate_broll_images(
    base_prompt,
    output_path,
    model_id=None,
    style_suffix=None,
    num_steps=None,
    guidance_scale=None,
):
    """
    Generates a single AI image and saves it.
    """
    model_id = model_id or os.environ.get("BROLL_MODEL_ID", "dreamlike-art/dreamlike-photoreal-2.0")
    pipe = load_pipeline(model_id)  # Load the pipeline (cached per model)

    # Ensure output directory exists
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        if any(x in model_id.lower() for x in ["waifu", "anime", "ghibli", "toon", "cartoon"]):
            default_suffix = ", Studio Ghibli style, anime illustration, painterly, soft lighting, vibrant colors"
        else:
            default_suffix = ", cinematic photo, photorealistic, 8k uhd, highly detailed, raw footage, masterpiece"

        style_suffix = style_suffix or os.environ.get("BROLL_STYLE_SUFFIX", default_suffix)
        full_prompt = base_prompt + style_suffix

        steps = num_steps if num_steps is not None else int(os.environ.get("BROLL_NUM_STEPS", "20"))
        guidance = guidance_scale if guidance_scale is not None else float(os.environ.get("BROLL_GUIDANCE_SCALE", "7.5"))

        # Stick to 512x512 for consistent vertical crop handling
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=512,
            height=512,
        ).images[0]

        image.save(output_path)
        flush()
        return output_path

    except Exception as exc:
        raise RuntimeError(f"Failed to generate/save image '{output_path}': {exc}")
