# def get_broll_prompts(transcript_segment):
#     prompt = f"""
#     Based on this transcript segment, identify 3 key visual moments to illustrate with AI images.
#     For each moment, provide:
#     1. The exact timestamp.
#     2. A highly detailed Stable Diffusion prompt (cinematic, 4k, digital art style).
    
#     Transcript: {transcript_segment}
    
#     Return ONLY a JSON list of objects:
#     [{{"time": 12.5, "prompt": "a futuristic city with neon lights, cinematic lighting"}}]
#     """
#     # Use the same ollama.chat logic from before...

import ollama, json, re

def get_creative_plan(transcript):
    prompt = f"""Analyze this transcript. 
    1. Find a 30s viral segment. 
    2. Suggest 3 AI image prompts for B-roll during that segment.
    Return ONLY JSON: {{"start": 0.0, "end": 30.0, "broll": [{{"time": 5.0, "prompt": "..."}}]}}
    Transcript: {transcript[:2000]}"""
    
    response = ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': prompt}])
    # Extract JSON using regex
    return json.loads(re.search(r'\{.*\}', response['message']['content'], re.DOTALL).group())