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
from diffusers import StableDiffusionXLPipeline 

# Add this BEFORE you load the pipeline
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

@st.cache_resource
def load_pipeline():
    # Setup device and data types
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    # Replace with your actual model ID (e.g., "Lykon/dreamshaper-xl-v2-turbo")
    model_id = "stabilityai/sdxl-turbo" 

    flush()
    print(f"--- Loading SDXL Turbo on {DEVICE.upper()} (Surgical Load) ---")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=DTYPE, 
        variant="fp16" if DEVICE == "cuda" else None
    )

    pipe.to(DEVICE)

    # Optional: Reduces memory usage on mid-range laptops
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

def generate_broll_images(base_prompt, output_path):
    """
    Generates a single AI image and saves it.
    """
    pipe = load_pipeline()  # Load the pipeline (cached for efficiency)
    # Ensure output directory exists
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        # SDXL Turbo is optimized for 1 step and 0 guidance
        # image = pipe(
        #     prompt=prompt, 
        #     num_inference_steps=1, 
        #     guidance_scale=0.0
        # ).images[0]
        
        # image.save(output_path)
        # return output_path

        # Add high-quality triggers to whatever Gemma 3 gave us
        style_suffix = ", cinematic photo, photorealistic, 8k uhd, highly detailed, raw footage, masterpiece"
        full_prompt = base_prompt + style_suffix
        
        # Stick to the 512x512 the repo recommends for best quality
        image = pipe(
            prompt=full_prompt, 
            num_inference_steps=1, 
            guidance_scale=0.0,
            width=512,
            height=512
        ).images[0]
        
        image.save(output_path)
        # 3. Clean up immediately after generation
        del pipe
        flush()
        return output_path

    except Exception as exc:
        raise RuntimeError(f"Failed to generate/save image '{output_path}': {exc}")
