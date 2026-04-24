import subprocess
import shutil
import time
import os
import sys

# # Replace 'tanve' with your actual Windows username if different
# env_path = r"C:\Users\tanve\Anaconda3\envs\py311"

# # Manually add the library paths to the Windows Search Path
# os.add_dll_directory(os.path.join(env_path, "Library", "bin"))
# os.add_dll_directory(os.path.join(env_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"))

# 1. Dynamically find your site-packages folder
env_path = sys.prefix 
nvidia_base = os.path.join(env_path, "Lib", "site-packages", "nvidia")

# 2. Add ALL NVIDIA bin folders to the DLL search path
if os.path.exists(nvidia_base):
    for root, dirs, files in os.walk(nvidia_base):
        if "bin" in dirs:
            bin_path = os.path.join(root, "bin")
            print(f"Adding DLL folder: {bin_path}")
            os.add_dll_directory(bin_path)

import torch

from utils.downloader import download_yt
from utils.transcriber import transcribe
from utils.highlights import get_viral_highlights
from utils.image_gen import generate_broll_images
from utils.visualizer import assemble_final_video

def check_environment():
    print("--- Checking Environment ---")
    
    # 1. Check FFmpeg
    if not shutil.which("ffmpeg"):
        # Fallback: Hardcode path if you haven't fixed the Windows Path yet
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found! Please ensure it's in C:\ffmpeg\bin")
    print("✅ FFmpeg is ready.")

    # 2. Check Ollama
    try:
        # Try to ping the local Ollama server
        import requests
        requests.get("http://localhost:11434")
    except:
        print("⚠️ Ollama not running. Attempting to start...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3) # Give it a moment to wake up
    print("✅ Ollama is ready.")

def main():
    try:
        check_environment() # Run our tool check first
        
        url = input("Paste YouTube URL: ")
        
        # Modular Pipeline
        video_file = download_yt(url)
        transcript = transcribe(video_file)
        
        viral_highlights = get_viral_highlights(transcript)
        print(f"Found viral highlights: {viral_highlights}")

        # Generate B-roll images for each highlight
        broll_images = []
        # for i, highlight in enumerate(viral_highlights):
        #     output_path = f"output/broll_{i}.png"
        #     generate_broll_images(highlight, output_path)
        #     broll_images.append(output_path)

        # 2. Loop through the cues and generate images
        for i, cue in enumerate(viral_highlights['broll']):
            image_filename = f"output/broll_{i}.png"
            print(f"Generating Image {i+1}/{len(viral_highlights['broll'])}: {cue['prompt'][:50]}...")
            
            path = generate_broll_images(cue['prompt'], image_filename)
            broll_images.append(path)
        
        print(f"Generated B-roll images: {broll_images}")
        print("🚀 Assembling final video with dynamic zoom and B-roll...")
        # Assemble final video with B-roll and dynamic zoom effects
        plan = {
            "start": 0,
            "end": 30, # Example duration
            "broll": [
                {"time": 5, "image": broll_images[0]},
                {"time": 15, "image": broll_images[1]},
                {"time": 25, "image": broll_images[2]}
            ]
        }
        final_video = assemble_final_video(video_file, plan, broll_images)
        
        print(f"🚀 Success! Your short is ready: {final_video}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()