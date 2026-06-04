import ast
import ollama
import json
import re

DEFAULT_HIGHLIGHT_COUNT = 5


def parse_time_to_seconds(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and val:
        return parse_time_to_seconds(val[0])
    if not isinstance(val, str):
        return float(val or 0.0)

    s = val.strip()
    if s.endswith('s'):
        s = s[:-1].strip()

    if ':' in s:
        parts = [float(p) if p else 0.0 for p in s.split(':')]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0.0
            minutes, seconds = parts
        else:
            return float(parts[0])
        return hours * 3600.0 + minutes * 60.0 + seconds

    try:
        return float(s)
    except ValueError:
        cleaned = ''.join(ch for ch in s if ch.isdigit() or ch == '.' or ch == '-')
        return float(cleaned or 0.0)


def format_time_for_prompt(val):
    seconds = parse_time_to_seconds(val)
    return f"{seconds:.1f}"


def extract_json_from_text(text):
    # Try extracting valid JSON from the AI response text.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'[\[\{]', text):
        start = match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue

    # Fallback: allow Python-style quotes/structures if JSON parsing fails.
    try:
        parsed = ast.literal_eval(text)
        return parsed
    except Exception:
        return None


def is_numeric(value):
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def is_valid_highlight_item(item, mode="tiktok"):
    if not isinstance(item, dict):
        return False
    if 'start' not in item or 'end' not in item or 'reason' not in item:
        return False
    if not is_numeric(item['start']) or not is_numeric(item['end']):
        return False
    start = parse_time_to_seconds(item['start'])
    end = parse_time_to_seconds(item['end'])
    # print(f"Validating highlight item: start={start}, end={end}")
    if start < 0 or end <= start or end - start > 180.0:
        return False
    if mode == 'clip' and (end - start < 15.0 or end - start > 90.0):
        return False
    # if mode != 'clip':
    #     if end - start < 30.0 or end - start > 60.0:
    #         return False
    if mode != 'clip':
        broll = item.get('broll')
        if broll is None or broll == []:
            return True
        if not isinstance(broll, list):
            return False
        if all(isinstance(cue, str) and cue.strip() for cue in broll):
            return True
        for cue in broll:
            if not isinstance(cue, dict):
                return False
            if 'prompt' not in cue:
                return False
            if 'time' in cue and not is_numeric(cue['time']):
                return False
            if not str(cue['prompt']).strip():
                return False
    return True


def is_valid_highlight_list(parsed, mode="tiktok"):
    if not isinstance(parsed, list) or len(parsed) < 3 or len(parsed) > DEFAULT_HIGHLIGHT_COUNT:
        return False
    return all(is_valid_highlight_item(item, mode=mode) for item in parsed)


def dedupe_highlights(highlights):
    unique = []
    seen = set()
    for item in highlights:
        key = (
            round(parse_time_to_seconds(item.get('start', 0.0)), 2),
            round(parse_time_to_seconds(item.get('end', 0.0)), 2),
            str(item.get('reason', '')).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def normalize_parsed_highlights(parsed, duration=None, mode="tiktok"):
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        wrapper_keys = ['highlights', 'data', 'result', 'items']
        for key in wrapper_keys:
            if key in parsed and isinstance(parsed[key], (list, dict)):
                parsed = parsed[key]
                break

    sanitized = normalize_highlight_list(parsed, duration=duration, mode=mode)
    sanitized = dedupe_highlights(sanitized)
    if len(sanitized) == 0:
        return None
    if len(sanitized) < 3:
        fallbacks = make_distinct_fallback_highlights(duration=duration, mode=mode, count=DEFAULT_HIGHLIGHT_COUNT - len(sanitized))
        sanitized.extend(fallbacks)
    if len(sanitized) > DEFAULT_HIGHLIGHT_COUNT:
        sanitized = sanitized[:DEFAULT_HIGHLIGHT_COUNT]
    if is_valid_highlight_list(sanitized, mode=mode):
        return sanitized
    return None


def maybe_convert_milliseconds(value, duration=None):
    if not isinstance(value, (int, float)):
        return value
    if duration and value > duration * 10 and value >= 1000:
        return value / 1000.0
    return value


def clamp_to_duration(value, duration):
    if duration is None:
        return value
    if value < 0:
        return 0.0
    return min(value, duration)


def sanitize_broll_cue(cue, clip_duration=None):
    if isinstance(cue, str):
        prompt = cue.strip()
        if not prompt:
            return None
        time = 0.0
        if clip_duration is not None and time > clip_duration:
            return None
        return {'time': time, 'prompt': prompt}
    if not isinstance(cue, dict):
        return None
    if 'time' not in cue:
        if 'cue_time' in cue:
            cue['time'] = cue['cue_time']
        elif 'cueTime' in cue:
            cue['time'] = cue['cueTime']
    time = parse_time_to_seconds(cue.get('time', 0.0))
    if clip_duration is not None and (time < 0 or time > clip_duration):
        return None
    prompt = str(cue.get('prompt', '')).strip()
    if not prompt:
        return None
    return {
        'time': time,
        'prompt': prompt,
    }


def generate_default_broll_cues(start, end, reason="", count=3):
    duration = max(0.1, float(end) - float(start))
    anchors = [0.2, 0.5, 0.8]
    base_reason = str(reason).strip() or "Highlight scene"
    prompts = [
        f"Cinematic visual representing the key moment: {base_reason}",
        f"Photorealistic scene illustrating the main idea of this segment.",
        f"Dynamic social media style image matching the tone of this highlight."
    ]
    cues = []
    for idx in range(count):
        cue_time = float(start) + min(duration, max(0.0, duration * anchors[idx]))
        cues.append({
            'time': float(min(float(end), cue_time)),
            'prompt': prompts[idx] if idx < len(prompts) else prompts[-1],
        })
    return cues


def sanitize_highlight_item(item):
    if not isinstance(item, dict):
        return {
            'start': 0.0,
            'end': 0.0,
            'reason': '',
            'broll': [],
        }
    if 'start_time' in item and 'start' not in item:
        item['start'] = item['start_time']
    if 'end_time' in item and 'end' not in item:
        item['end'] = item['end_time']
    if 'startTime' in item and 'start' not in item:
        item['start'] = item['startTime']
    if 'endTime' in item and 'end' not in item:
        item['end'] = item['endTime']
    if 'start_timestamp' in item and 'start' not in item:
        item['start'] = item['start_timestamp']
    if 'end_timestamp' in item and 'end' not in item:
        item['end'] = item['end_timestamp']

    if 'broll' not in item:
        if 'b_roll' in item:
            item['broll'] = item['b_roll']
        elif 'bRoll' in item:
            item['broll'] = item['bRoll']
        elif 'b-roll' in item:
            item['broll'] = item['b-roll']

    broll = item.get('broll', [])
    if not isinstance(broll, list):
        broll = []

    return {
        'start': parse_time_to_seconds(item.get('start', 0.0)),
        'end': parse_time_to_seconds(item.get('end', 0.0)),
        'reason': str(item.get('reason', '')).strip(),
        'broll': [cue for cue in (sanitize_broll_cue(c) for c in broll) if cue is not None],
    }


def normalize_highlight_item(item, duration=None, mode="tiktok"):
    normalized = sanitize_highlight_item(item)
    start = maybe_convert_milliseconds(normalized['start'], duration)
    end = maybe_convert_milliseconds(normalized['end'], duration)

    if duration is not None:
        start = clamp_to_duration(start, duration)
        end = clamp_to_duration(end, duration)
        if start >= duration:
            start = max(0.0, duration - min(30.0, duration))
        if end >= duration:
            end = max(start + 0.1, min(duration - 1e-3, end))

    if end <= start:
        if duration is not None:
            end = duration
            start = max(0.0, duration - min(90.0, duration))
        else:
            end = start + 30.0

    min_length = 15.0 if mode == 'clip' else 30.0
    max_length = 90.0 if mode == 'clip' else 60.0
    if end - start > max_length:
        end = start + max_length
        if duration is not None and end > duration:
            end = duration
            start = max(0.0, end - max_length)

    if end - start < min_length:
        end = max(end, start + min_length)
        if duration is not None and end > duration:
            end = duration
            start = max(0.0, end - min_length)

    normalized['start'] = float(max(0.0, start))
    normalized['end'] = float(max(normalized['start'] + 0.1, end))
    normalized['reason'] = str(normalized.get('reason', '')).strip() or 'AI-selected highlight segment.'
    if mode == 'clip':
        normalized['broll'] = []
    else:
        valid_broll = [
            cue for cue in (sanitize_broll_cue(cue, clip_duration=normalized['end'] - normalized['start']) for cue in normalized.get('broll', []))
            if cue is not None
        ]
        if len(valid_broll) >= 3:
            normalized['broll'] = valid_broll[:3]
        else:
            default_broll = generate_default_broll_cues(normalized['start'], normalized['end'], normalized['reason'], count=3)
            for idx, cue in enumerate(valid_broll):
                default_broll[idx] = cue
            normalized['broll'] = default_broll
    return normalized


def make_fallback_highlight(start=0.0, duration=None, mode='tiktok'):
    clip_length = 30.0 if mode == 'clip' else min(45.0, max(15.0, (duration or 60.0) / 3.0))
    if duration is not None:
        clip_length = min(clip_length, duration)
    start = max(0.0, min(start, max(0.0, (duration or 0.0) - clip_length)))
    end = float(max(start + 1.0, min(start + clip_length, duration if duration is not None else start + clip_length)))
    return {
        'start': float(start),
        'end': end,
        'reason': 'Fallback highlight due to invalid AI output.',
        'broll': [] if mode == 'clip' else [],
    }


def make_distinct_fallback_highlights(duration=None, mode='tiktok', count=3):
    if duration is None or duration <= 0:
        return [make_fallback_highlight(0.0, duration, mode) for _ in range(count)]

    clip_length = 30.0 if mode == 'clip' else min(45.0, max(15.0, duration / 3.0))
    clip_length = min(clip_length, duration)
    if count == 1:
        starts = [0.0]
    else:
        available_span = max(0.0, duration - clip_length)
        step = available_span / max(1, count - 1)
        starts = [round(i * step, 3) for i in range(count)]

    return [make_fallback_highlight(start, duration, mode) for start in starts]


def normalize_highlight_list(parsed, duration=None, mode='tiktok'):
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [make_fallback_highlight(duration=duration, mode=mode) for _ in range(DEFAULT_HIGHLIGHT_COUNT)]

    sanitized = [normalize_highlight_item(item, duration=duration, mode=mode) for item in parsed]
    sanitized = [item for item in sanitized if item['end'] > item['start']]
    sanitized = dedupe_highlights(sanitized)
    if not sanitized:
        return make_distinct_fallback_highlights(duration=duration, mode=mode, count=DEFAULT_HIGHLIGHT_COUNT)

    if len(sanitized) < 3:
        fallbacks = make_distinct_fallback_highlights(duration=duration, mode=mode, count=DEFAULT_HIGHLIGHT_COUNT - len(sanitized))
        sanitized.extend(fallbacks)

    return sanitized[:DEFAULT_HIGHLIGHT_COUNT]


def get_viral_highlights(transcript_data, mode="tiktok", video_duration=None):
    """
    Sends the transcript to Gemma 4 and gets back start/end timestamps.
    """
    full_text = "\n".join([
        f"[{format_time_for_prompt(item['start'])} - {format_time_for_prompt(item['end'])}]: {item['text']}"
        for item in transcript_data
    ])

    if mode == "clip":
        duration_text = f"\nTOTAL VIDEO DURATION: {video_duration:.1f} seconds\n" if video_duration is not None else ""
        prompt = f"""
        IMPORTANT: Respond ONLY in English. Do not use any other language.
        Use only numeric timestamps in seconds, without letter suffixes.
        You are an expert trailer editor. Your task is to extract three strong clips from the transcript that work as Spotify/YouTube trailer previews.

        TRANSCRIPT TO ANALYZE:
        {full_text}
        {duration_text}

        INSTRUCTIONS:
        1. SELECT THREE DIFFERENT CLIPS: Identify three different trailer-worthy segments.
           Each clip must be between 15 and 90 seconds long, inclusive.
        2. DO NOT RETURN A CLIP LONGER THAN 90 SECONDS.
        3. Each clip must start at or after 0.0 and end at or before the total video duration.
        4. AVOID MUSIC INTROS AND FILLER: Do not pick a timestamp range that is only intro music, title announcements, or silence.
        5. KEEP IT NATURAL: Choose segments with clear spoken content, strong story beats, or memorable moments.
        6. Provide a short "reason" explaining why this segment works as a trailer preview.

        RULES:
        - Return ONLY a JSON array with 3 to 5 objects.
        - Each object must have keys: "start", "end", "reason".
        - Each "start" and "end" value must be numeric seconds within the video duration.
        - Do not return any extra fields, including topic, description, keywords, or analysis.

        JSON STRUCTURE:
        [
        {{"start": 0.0, "end": 0.0, "reason": "..."}},
        {{"start": 0.0, "end": 0.0, "reason": "..."}},
        {{"start": 0.0, "end": 0.0, "reason": "..."}}
        ]
        Return EXACTLY the JSON array and nothing else.
        """
    else:
        duration_text = f"\nTOTAL VIDEO DURATION: {video_duration:.1f} seconds\n" if video_duration is not None else ""
        prompt = f"""
        IMPORTANT: Respond ONLY in English. Do not use any other language.
        Use only numeric timestamps in seconds, without letter suffixes.
        You are an expert viral social media editor. Your task is to extract high-engagement segments from the transcript below.
        Find THREE different engaging 30-60 second segments that work as standalone TikTok Shorts. Look for: A strong opening hook, a clear point, and a satisfying ending.

        TRANSCRIPT TO ANALYZE:
        {full_text}
        {duration_text}

        INSTRUCTIONS:
        1. SELECT THREE DIFFERENT CLIPS: Identify three different "viral" 30-60 second segments.
           Each clip should start and end at valid timestamps found in the transcript.
           Make sure each clip length is between 30 and 60 seconds. Do not return any clip shorter than 30 seconds or longer than 60 seconds.
        2. DO NOT RETURN A CLIP LONGER THAN 60 SECONDS.
        3. Each clip must start at or after 0.0 and end at or before the total video duration.
        4. VARIETY: The three clips should be different, capturing different moments or angles of the content.
        5. B-ROLL CUES: For each clip, select 3 moments INSIDE that clip for AI-generated visuals.
           Use the field name exactly "broll" (lowercase, no underscores).
           Each broll entry must be an object with exactly two keys: "time" and "prompt".
           Do not use "b_start", "b_end", "b_roll", "bRoll", or "b-roll" or any other field names. JUST broll.
        6. PROMPT STYLE: Prompts must be photorealistic, cinematic, and suitable for short-form social media.

        RULES:
        - Return ONLY a JSON array with 3 to 5 objects.
        - Each object must have the keys: "start", "end", "reason", "broll".
        - Use actual timestamps from the transcript provided.
        - Each "start" and "end" must be numeric seconds within the video duration.
        - Each "broll" list must contain exactly 3 cue objects.
        - Do not use any other field names for b-roll (such as b_roll, bRoll, or b-roll).
        - Name the field exactly "broll" and do not use "b_roll", "bRoll", or "b-roll".

        JSON STRUCTURE:
        [
        {{
            "start": 0.0,
            "end": 0.0,
            "reason": "...",
            "broll": [
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}}
            ]
        }},
        {{
            "start": 0.0,
            "end": 0.0,
            "reason": "...",
            "broll": [
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}}
            ]
        }},
        {{
            "start": 0.0,
            "end": 0.0,
            "reason": "...",
            "broll": [
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}},
                {{"time": 0.0, "prompt": "..."}}
            ]
        }}
        ]
        Do not return any other fields, including topic, description, or keywords.
        Return EXACTLY the JSON array and nothing else.
        """

    print("--- Consulting Gemma 4 for the top 3 viral moments ---")
    max_attempts = 3
    last_parsed = None
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n\nThe previous response was malformed. "
                "You must return EXACTLY the JSON array with 3 to 5 objects using only 'start', 'end', and 'reason' "
                "for clip mode, or 'start', 'end', 'reason', and 'broll' for TikTok mode. "
                "Do not add any extra fields or explanatory text. "
                "Each TikTok clip must be between 30 and 60 seconds. "
                "Each clip mode clip must be between 15 and 90 seconds. "
                "All start/end values must be within the total video duration. "
                "For TikTok mode, the field name must be exactly 'broll'. NOT 'b_roll', 'bRoll', or 'b-roll' or any other variation. "
                "Do not use any other field names for b-roll, including b_start, b_end, b_roll, bRoll, or b-roll."
            )

        response = ollama.chat(model='gemma4:e4b', messages=[
            {'role': 'user', 'content': attempt_prompt},
        ],
        options={
            'num_gpu': 99
        })
        content = response['message']['content']
        # print(f"Raw AI response (attempt {attempt}): {content}")

        parsed = extract_json_from_text(content)
        last_parsed = parsed
        if parsed is None:
            print(f"Attempt {attempt}: No valid JSON found.")
            continue

        normalized = normalize_parsed_highlights(parsed, duration=video_duration, mode=mode)
        if normalized is not None:
            if not is_valid_highlight_list(parsed, mode=mode):
                print(f"Attempt {attempt}: Parsed JSON was malformed but auto-normalized successfully.")
            return normalized

        print(f"Attempt {attempt}: JSON schema invalid or values incorrect.")
        continue

    print("Failed to get valid Gemma output after retries.")
    if last_parsed is not None:
        sanitized = normalize_highlight_list(last_parsed, duration=video_duration, mode=mode)
        print("Using normalized highlights from last parsed output.")
        return sanitized
    print("Using fallback highlights because no JSON response could be parsed.")
    return make_distinct_fallback_highlights(duration=video_duration, mode=mode, count=3)

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