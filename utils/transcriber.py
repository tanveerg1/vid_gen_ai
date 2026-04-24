from faster_whisper import WhisperModel

def transcribe(file_path):
    # 'base' model is perfect for clear audio-only or static videos
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(file_path)
    
    transcript_data = []
    for s in segments:
        transcript_data.append({"start": s.start, "end": s.end, "text": s.text})
    return transcript_data

import pandas as pd

def get_transcript_df(video_file):
    # This calls your existing transcribe logic
    transcript_data = transcribe(video_file) 
    # Convert list of dicts: [{'start': 0.0, 'end': 2.0, 'text': 'Hello'}, ...]
    return pd.DataFrame(transcript_data)