from faster_whisper import WhisperModel

# def transcribe(file_path):
    # # 'base' model is perfect for clear audio-only or static videos
    # model = WhisperModel("base", device="cpu", compute_type="int8")
    # segments, _ = model.transcribe(file_path)
    
    # transcript_data = []
    # for s in segments:
        # transcript_data.append({"start": s.start, "end": s.end, "text": s.text})
    # return transcript_data
    
def transcribe(file_path):
    # 'base' model is perfect for clear audio-only or static videos
    model = WhisperModel("base", device="cpu", compute_type="int8")
    # CRITICAL: Tell the model to extract word-level timestamps
    segments, _ = model.transcribe(file_path, word_timestamps=True)
    
    transcript_data = []
    for s in segments:
        segment_dict = {"start": s.start, "end": s.end, "text": s.text}
        
        # Capture the raw word timestamps from the segment
        if s.words:
            segment_dict["words"] = [
                {"start": w.start, "end": w.end, "text": w.word} 
                for w in s.words
            ]
            
        transcript_data.append(segment_dict)
    return transcript_data

import pandas as pd

def get_transcript_df(video_file):
    # This calls your existing transcribe logic
    transcript_data = transcribe(video_file) 
    # Convert list of dicts: [{'start': 0.0, 'end': 2.0, 'text': 'Hello'}, ...]
    return pd.DataFrame(transcript_data)