import ollama
import json
import re

def get_viral_highlights(transcript_data):
    """
    Sends the transcript to Gemma 3 and gets back start/end timestamps.
    """
    # # Convert our list of dicts into one big string for the AI
    # full_text = "\n".join([f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in transcript_data])
    
    # prompt = f"""
    # You are an expert social media editor. Analyze this transcript from a static-image video.
    # Find the most engaging 30-60 second segment that works as a standalone YouTube Short.
    # Look for: A strong opening hook, a clear point, and a satisfying ending.
    
    # Transcript:
    # {full_text}
    
    # RESPONSE FORMAT:
    # You must return ONLY a JSON object with this exact structure:
    # {{"start": 0.0, "end": 0.0, "reason": "why this is the hook"}}
    # """

    full_text = "\n".join([f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in transcript_data])
    
    prompt = f"""
    IMPORTANT: Respond ONLY in English. Do not use any other language. Do not add s for seconds in your response. Only return the numbers.
    You are an expert viral social media editor. Your task is to extract high-engagement segments from the transcript below.
    Find THREE different engaging 30-60 second segments that work as standalone TikTok Shorts. Look for: A strong opening hook, a clear point, and a satisfying ending.

    TRANSCRIPT TO ANALYZE:
    {full_text}

    INSTRUCTIONS:
    1. SELECT THREE DIFFERENT CLIPS: Identify three different "viral" 30-60s segments. Each must start and end at valid timestamps found in the transcript. Make sure each clip length is between 30 and 60 seconds.
    2. VARIETY: The three clips should be different, capturing different moments or angles of the content.
    3. B-ROLL CUES: For each clip, pick 3 moments INSIDE that clip for AI-generated visuals.
    4. PROMPT STYLE: Prompts must be photorealistic, 8k, and cinematic.

    RULES:
    - Do NOT use the numbers from the example below.
    - Use ACTUAL timestamps from the transcript provided.
    - Return ONLY a JSON array with 3 objects.
    - Each object must have the same structure.

    JSON STRUCTURE (Use this as a schema, not for values):
    [
    {{
    "start": [insert actual start time from transcript],
    "end": [insert actual end time from transcript],
    "reason": "Explain why this specific hook works",
    "broll": [
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }}
    ]
    }},
    {{
    "start": [insert actual start time from transcript],
    "end": [insert actual end time from transcript],
    "reason": "Explain why this specific hook works",
    "broll": [
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }}
    ]
    }},
    {{
    "start": [insert actual start time from transcript],
    "end": [insert actual end time from transcript],
    "reason": "Explain why this specific hook works",
    "broll": [
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }},
        {{
        "time": [timestamp between start and end], 
        "prompt": "detailed cinematic prompt"
        }}
    ]
    }}
    ]
    """
    
    print("--- Consulting Gemma 3 for the top 3 viral moments ---")
    response = ollama.chat(model='gemma4:e4b', messages=[
        {'role': 'user', 'content': prompt},
    ])
    # print(f"AI response received: {response}")
    # Extract the JSON from the AI's response
    content = response['message']['content']
    print(f"Raw AI response: {content}")
    try:
        # Use regex to find the JSON array block if the AI added extra text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            # Ensure we have an array
            if isinstance(parsed, list):
                # If we got fewer than 3, duplicate the first one to fill the gaps
                while len(parsed) < 3:
                    parsed.append(parsed[0])
                # Return only the first 3 if somehow more were returned
                return parsed[:3]
            else:
                # Single object returned, use it to create 3 copies
                print("Warning: Got single highlight instead of 3, duplicating...")
                return [parsed, parsed, parsed]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            while len(parsed) < 3:
                parsed.append(parsed[0])
            return parsed[:3]
        else:
            return [parsed, parsed, parsed]
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return None

# --- QUICK TEST ---
if __name__ == "__main__":
    # Dummy data to test the connection to Ollama
    test_transcript = [
        {"start": 0.0, "end": 10.0, "text": "Welcome to my podcast about space."},
        {"start": 10.0, "end": 45.0, "text": "The most insane thing about black holes is that they actually warp time itself! Imagine being stuck in a moment forever."},
        {"start": 45.0, "end": 60.0, "text": "Thanks for listening to this episode."}
    ]
    result = get_viral_highlights(test_transcript)
    print(f"AI Selected: {result}")