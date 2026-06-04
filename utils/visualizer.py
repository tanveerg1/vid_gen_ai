from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# Import Resize effect (MoviePy 2.1.2+ uses class-based effects)
try:
    from moviepy.video.fx.Resize import Resize
except Exception:
    Resize = None

import os
import cv2
import numpy as np
import traceback


def resize_clip(clip, **kwargs):
    """Resize a clip using MoviePy 2.1.2+ with_effects API."""
    if 'newsize' in kwargs:
        size = kwargs.pop('newsize')
        if isinstance(size, tuple) and len(size) == 2:
            kwargs['width'], kwargs['height'] = size
        else:
            kwargs['width'] = size

    if Resize is None:
        if hasattr(clip, 'resize'):
            return clip.resize(**kwargs)
        raise AttributeError('Resize effect not available in this MoviePy version')
    
    # Use with_effects for MoviePy 2.1.2+
    if hasattr(clip, 'with_effects'):
        return clip.with_effects([Resize(**kwargs)])
    
    # Fallback for older versions
    if hasattr(clip, 'fx'):
        return clip.fx(Resize, **kwargs)
    
    raise AttributeError('Clip does not support with_effects or fx methods')


def safe_clip_end(end, duration, epsilon=1e-3):
    """Return a safe end time inside the video duration to avoid last-frame boundary issues."""
    if duration is None:
        return end
    if end > duration:
        end = duration
    end = min(end, max(0.0, duration - epsilon))
    return end

def get_face_x_center(video_path: str, target_width_ratio: float = 9/16) -> float:
    """
    Detects the primary face in a video and returns an optimal x-center for cropping
    that keeps the face fully visible within the crop, accounting for face width.
    
    Args:
        video_path: Path to the video file
        target_width_ratio: The width ratio of the target crop (width/height), default 9:16
        
    Returns:
        Optimal x-center position (0.0 to 1.0) that fits the entire face in the crop.
        Returns 0.5 if no face is detected (center fallback).
    """
    try:
        with VideoFileClip(video_path) as video_clip:
            width, height = video_clip.size
            duration = video_clip.duration
            
            # Calculate crop width based on height and target ratio
            crop_width = height * target_width_ratio
            crop_width_norm = crop_width / width  # Normalized to 0-1 range
            
            # Sample frames throughout the video to find faces
            sample_times = np.linspace(0, duration * 0.9, min(10, int(duration * 2)))
            face_bounds = []  # Store (left, right) in normalized coordinates
            
            for sample_time in sample_times:
                try:
                    # Try to get frame - handle different MoviePy versions
                    frame = None
                    if hasattr(video_clip, 'get_frame'):
                        frame = video_clip.get_frame(sample_time)
                    elif hasattr(video_clip, 'get_frame'):
                        frame = video_clip.get_frame(sample_time)
                    else:
                        print(f"Cannot get frame from video_clip (no get_frame method)")
                        continue
                    
                    if frame is None:
                        continue
                        
                    detected_faces = detect_faces_in_frame(frame, width, height)
                    
                    if detected_faces:
                        # Get the largest/most prominent face
                        largest_face = max(detected_faces, key=lambda f: f[2] * f[3])
                        x, y, w, h, confidence = largest_face
                        
                        # Calculate face left and right edges in normalized coordinates
                        face_left = x / width
                        face_right = (x + w) / width
                        face_bounds.append((face_left, face_right))
                        
                except Exception as e:
                    print(f"Error processing frame at {sample_time}s: {e}")
                    continue
            
            if face_bounds:
                # Use median bounds to be robust against outliers
                face_lefts = [b[0] for b in face_bounds]
                face_rights = [b[1] for b in face_bounds]
                median_face_left = np.median(face_lefts)
                median_face_right = np.median(face_rights)
                
                face_width = median_face_right - median_face_left
                half_crop = crop_width_norm / 2
                
                if face_width < crop_width_norm:
                    # Face fits in crop, center on face center
                    face_center = (median_face_left + median_face_right) / 2
                    x_center = face_center
                else:
                    # Face too wide, use logic to keep entire face visible
                    
                    min_x_center = median_face_left + half_crop
                    max_x_center = median_face_right - half_crop
                    
                    if max_x_center >= min_x_center:
                        x_center = (min_x_center + max_x_center) / 2
                    else:
                        x_center = min_x_center
                
                # Clamp to valid range to avoid cropping beyond video edges
                min_valid = half_crop
                max_valid = 1 - half_crop
                x_center = max(min_valid, min(max_valid, x_center))
                
                # print(f"Face detected: x_center={x_center:.3f}")
                return float(x_center)
            else:
                # Default to center if no face detected
                print(f"No faces detected in video, using center position")
                print(f"No faces detected in video, using center position")
                return 0.5
    except Exception as e:
        print(f"Error in get_face_x_center: {e}")
        traceback.print_exc()
        return 0.5

def detect_faces_in_frame(frame, width, height):
    """
    Helper function to detect faces in a single frame.
    Returns list of (x, y, w, h, confidence) tuples.
    """
    detected_faces = []
    
    try:
        # Try MediaPipe first
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
        
        results = mp_face_detection.process(frame)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                if w > 30 and h > 30:
                    detected_faces.append((x, y, w, h, confidence))
        
        mp_face_detection.close()
        
    except (ImportError, Exception):
        # Fallback to OpenCV Haar Cascade
        try:
            haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            faces = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(40, 40),
                maxSize=(int(width * 0.7), int(height * 0.7)),
            )
            
            for x, y, w, h in faces:
                face_area = w * h
                frame_area = width * height
                relative_area = face_area / frame_area
                
                if 0.005 < relative_area < 0.3:
                    confidence = min(0.9, 0.3 + (relative_area * 2))
                    detected_faces.append((x, y, w, h, confidence))
                    
        except Exception:
            pass
    
    return detected_faces

def create_karaoke_caption(word_text, duration, main_size, is_active=False):
    """Creates the text layer. Active words are Green and slightly larger."""
    color = '#00FF00' if is_active else '#FFFFFF'
    # Same size for active to avoid overlay issues
    size_mult = 1.0
    
    txt = TextClip(
        text=word_text, # upper()
        font_size=int(35 * size_mult), 
        color=color,
        font=r"C:\Windows\Fonts\arialbd.ttf",
        stroke_color='black',
        stroke_width=4,
        method='caption',
        size=(int(main_size[0] * 0.95), 60)  # Limit height to prevent cutoff
    ).with_duration(duration)
    
    return txt


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

    return float(s)


def flatten_transcript_words(transcript_data, clip_start=0.0):
    """Convert transcript entries into word-level timestamps."""
    words = []
    # print(f"transcript_data: {transcript_data}")
    for entry in transcript_data:
        entry_text = str(entry.get("text", "")).strip()
        if not entry_text:
            continue 

        entry_start = parse_time_to_seconds(entry.get("start", 0.0)) - parse_time_to_seconds(clip_start)
        entry_end = parse_time_to_seconds(entry.get("end", entry_start)) - parse_time_to_seconds(clip_start)
        entry_duration = max(0.0, entry_end - entry_start)

        if entry.get("words"):
            for word in entry["words"]:
                word_text = str(word.get("text", "")).strip()
                if not word_text:
                    continue
                word_start = parse_time_to_seconds(word.get("start", entry_start)) - parse_time_to_seconds(clip_start)
                word_end = parse_time_to_seconds(word.get("end", word_start)) - parse_time_to_seconds(clip_start)
                words.append({"word": word_text, "start": word_start, "end": word_end})
            continue

        tokens = entry_text.split()
        if len(tokens) == 1 or entry_duration <= 0:
            words.append({"word": entry_text, "start": entry_start, "end": entry_end})
            continue

        part_duration = entry_duration / len(tokens)
        for idx, token in enumerate(tokens):
            word_start = entry_start + idx * part_duration
            word_end = entry_start + (idx + 1) * part_duration
            words.append({"word": token, "start": word_start, "end": word_end})

    return words

def create_karaoke_word_group_clips(
    word_group,
    clip_size,
    font_path,
    font_size,
    normal_color,
    highlight_color,
    total_group_bounds,
    stroke_width=4,
    position_y=0.68,
):
    clips = []
    group_text = " ".join(word["word"] for word in word_group)
    
    ## new code
    group_start, group_end = total_group_bounds
    ####
    ## old working code
    # group_start = word_group[0]["start"]
    # group_end = max(word["end"] for word in word_group)
    group_duration = max(0.1, group_end - group_start)

    # 1. Shared safe height to prevent cutoff (2x font size is very safe)
    safe_height = int(font_size * 2.0)
    
    # 2. Base Clip (Normal Color)
    base_clip = TextClip(
        text=group_text,
        font=font_path,
        font_size=font_size,
        color=normal_color,
        stroke_color="black",
        stroke_width=stroke_width,
        method="label",
        size=(None, safe_height),
    ).with_start(group_start).with_duration(group_duration)

    ### new code
    vertical_position = int(clip_size[1] * position_y) - (safe_height // 2)
    ####
    # Position logic
    # vertical_position = int(clip_size[1] - safe_height - 250) #old code
    base_clip = base_clip.with_position(("center", vertical_position))
    
    full_width = base_clip.size[0]
    clips.append(base_clip)

    # 3. Create Highlight Clips via Cropping
    current_text_accumulator = ""
    
    for word in word_group:
        word_start = max(0, word["start"])
        word_end = max(word_start + 0.05, word["end"])
        word_duration = max(0.05, word_end - word_start)

        # Helper to get precise width of string segments
        def get_width(txt):
            if not txt: return 0
            # Use same params as base_clip for identical kerning
            with TextClip(text=txt, font=font_path, font_size=font_size, 
                          method="label", stroke_width=stroke_width) as t_clip:
                return t_clip.size[0]

        buffer = 4  # Pixels to expand the crop to cover the stroke/edges

        x1 = max(0, get_width(current_text_accumulator) - buffer)
        # Add the word + buffer for x2
        x2 = min(full_width, get_width(current_text_accumulator + word["word"]) + buffer)
        
        # Create the full-sentence highlight clip
        full_highlight = TextClip(
            text=group_text,
            font=font_path,
            font_size=font_size,
            color=highlight_color,
            stroke_color="black",
            stroke_width=stroke_width,
            method="label",
            size=(full_width, safe_height),
        ).with_start(word_start).with_duration(word_duration)

        # MOVIEPY 2.0: Use the expanded x1 and x2
        highlight_word_only = full_highlight.cropped(x1=x1, y1=0, x2=x2, y2=safe_height)
        
        # Position logic: We MUST use the same x1 used in the crop 
        # to ensure the word stays pinned to the correct background pixels
        start_x_of_base = (clip_size[0] - full_width) // 2
        highlight_word_only = highlight_word_only.with_position((int(start_x_of_base + x1), vertical_position))
        
        # --- NEW: POP/ZOOM EFFECT ---
        # zoom_factor = 1.2  # 20% bigger
        
        # # Resize the cropped word
        # # In MoviePy 2.0, resized takes a float or (width, height)
        # highlight_word_only = highlight_word_only.resized(zoom_factor)
        
        # # Calculate the center-alignment offset
        # # When we scale, the clip grows. To keep it centered, we shift it:
        # # offset = (Original Size * (Zoom - 1)) / 2
        # extra_w = ( (x2 - x1) * (zoom_factor - 1) ) / 2
        # extra_h = ( safe_height * (zoom_factor - 1) ) / 2
        
        # # Final Position calculation
        # start_x_of_base = (clip_size[0] - full_width) // 2
        
        # # Subtract the extra_w and extra_h to keep it centered over the white text
        # final_x = int(start_x_of_base + x1 - extra_w)
        # final_y = int(vertical_position - extra_h)

        # highlight_word_only = highlight_word_only.with_position((final_x, final_y))
        # -----------------------------

        clips.append(highlight_word_only)
        
        # Update accumulator for the next word calculation
        current_text_accumulator += word["word"] + " "

    return clips



def create_fast_preview(input_path, start, end, transcript_data, output_path="output/preview_highlight.mp4", render_mode="tiktok"):
    """Creates a preview using MoviePy v2.0+ syntax."""
    if not os.path.exists("output"):
        os.makedirs("output")
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)

    try:
        with VideoFileClip(abs_input) as video:
            start = float(start)
            end = float(end)
            if start < 0:
                start = 0.0
            if end < 0:
                end = 0.0
            if start >= video.duration:
                print(f"Invalid preview range: start {start} >= video duration {video.duration}")
                return None
            end = safe_clip_end(end, video.duration)
            if end <= start:
                print(f"Invalid preview range after clamping: end {end} <= start {start}")
                return None

            new_clip = video.subclipped(start, end)

            if render_mode == "tiktok":
                # Crop to vertical (9:16) like the final video
                w, h = new_clip.size
                target_ratio = 9 / 16
                new_w = h * target_ratio
                new_clip = new_clip.cropped(x_center=get_face_x_center(abs_input) * w, width=new_w)
                # Force true HD vertical output for TikTok/Reels/Shorts
                new_clip = resize_clip(new_clip, width=1080, height=1920)

            if new_clip.size[0] < 768:
                new_clip = resize_clip(new_clip, width=768)
            elif new_clip.size[0] % 2 != 0:
                # Force width to even if it somehow became odd
                new_clip = resize_clip(new_clip, width=new_clip.size[0] + 1)
                
            if new_clip.size[1] < 1024:
                new_clip = resize_clip(new_clip, height=1024)
            elif new_clip.size[1] % 2 != 0:
                # Force width to even if it somehow became odd
                new_clip = resize_clip(new_clip, width=new_clip.size[1] + 1)

            elements = [new_clip]

            if render_mode == "tiktok" and transcript_data:
                all_words = flatten_transcript_words(transcript_data, float(start))
                valid_words = []
                for word in all_words:
                    ### NEW CODE
                    # Clear boundary check: make sure words stay strictly inside the clip timeline
                    if word["end"] <= 0 or word["start"] >= (end - start):
                        continue
                    word["end"] = min((end - start), word["end"])
                    if word["end"] > word["start"]:
                        valid_words.append(word)
                    ###
                    
                    ### OLD WORKING CODE
                    # if word["end"] <= 0 or word["start"] >= new_clip.duration:
                        # continue
                    # word["start"] = max(0, word["start"])
                    # word["end"] = min(new_clip.duration, word["end"])
                    # if word["end"] > word["start"]:
                        # valid_words.append(word)
                    #####
                
                dynamic_font_size = int(new_clip.size[1] * 0.0365)
                
                # new fix for captions
                word_groups = []
                current_group = []
                
                for word in valid_words:
                    if not current_group:
                        current_group.append(word)
                    else:
                        # Start a new block if the sentence stretches over 1.6 seconds
                        group_duration = word["end"] - current_group[0]["start"]
                        if group_duration > 1.6 or len(current_group) >= 6:
                            word_groups.append(current_group)
                            current_group = [word]
                        else:
                            current_group.append(word)
                if current_group:
                    word_groups.append(current_group)
                    
                # Generate clips from our smart time-synced blocks
                for word_group in word_groups:
                    if not word_group:
                        continue
                    
                    # Calculate overall group bounds so both lines share them
                    group_start = word_group[0]["start"]
                    group_end = max(w["end"] for w in word_group)
                    bounds = (group_start, group_end)
                    
                    # Split the word group into 2 distinct lines for TikTok style layout
                    # midpoint = (len(word_group) + 1) // 2
                    # line1_words = word_group[:midpoint]
                    # line2_words = word_group[midpoint:]
                    
                    # Group midpoint time to separate line 1 and line 2 naturally
                    group_midtime = group_start + ((group_end - group_start) / 2)
                    
                    # Distribute words dynamically based on WHEN they are spoken
                    line1_words = [w for w in word_group if w["start"] < group_midtime]
                    line2_words = [w for w in word_group if w["start"] >= group_midtime]

                    # Render Line 1 (Slightly higher up)
                    if line1_words:
                        line1_display = [w.copy() for w in line1_words]
                        
                        elements.extend(
                            create_karaoke_word_group_clips(
                                line1_display,
                                new_clip.size,
                                r"C:\Windows\Fonts\arialbd.ttf",
                                dynamic_font_size,
                                "#FFFFFF",
                                "#00FF00",
                                total_group_bounds=bounds,
                                stroke_width=4,
                                position_y=0.68, # Line 1 position
                            )
                        )
                    
                    # Render Line 2 (Slightly lower down)
                    if line2_words:
                        line2_display = [w.copy() for w in line2_words]
                        
                        elements.extend(
                            create_karaoke_word_group_clips(
                                line2_display,
                                new_clip.size,
                                r"C:\Windows\Fonts\arialbd.ttf",
                                dynamic_font_size,
                                "#FFFFFF",
                                "#00FF00",
                                total_group_bounds=bounds,
                                stroke_width=4,
                                position_y=0.72, # Line 2 position
                            )
                        )
                
                #####
                
                
                #### old caption code
                # for group_start_idx in range(0, len(valid_words), 4):
                    # word_group = valid_words[group_start_idx : group_start_idx + 4]
                    # if not word_group:
                        # continue
                    # elements.extend(
                        # create_karaoke_word_group_clips(
                            # word_group,
                            # new_clip.size,
                            # r"C:\Windows\Fonts\arialbd.ttf",
                            # dynamic_font_size,
                            # "#FFFFFF",
                            # "#00FF00",
                            # stroke_width=4,
                            # #position_y=0.80, #old one
                            # position_y=0.68, # new test
                        # )
                    # )

            #final_clip = CompositeVideoClip(elements)
            final_clip = CompositeVideoClip(elements, size=new_clip.size).with_duration(new_clip.duration)
            
            has_audio = new_clip.audio is not None and getattr(new_clip.audio, 'duration', 0) > 0
            if has_audio:
                #new changes
                clip_audio = new_clip.audio.subclipped(0, new_clip.duration)
                if hasattr(final_clip, 'set_audio'):
                    final_clip = final_clip.set_audio(clip_audio)
                elif hasattr(final_clip, 'with_audio'):
                    final_clip = final_clip.with_audio(clip_audio)
                else:
                    final_clip.audio = new_clip.audio
                ####
                
                #if hasattr(final_clip, 'set_audio'):
                #    final_clip = final_clip.set_audio(new_clip.audio)
                #elif hasattr(final_clip, 'with_audio'):
                #    final_clip = final_clip.with_audio(new_clip.audio)
                #else:
                #    final_clip.audio = new_clip.audio
            else:
                if hasattr(final_clip, 'set_audio'):
                    final_clip = final_clip.set_audio(None)
                elif hasattr(final_clip, 'with_audio'):
                    final_clip = final_clip.with_audio(None)
                else:
                    final_clip.audio = None
            final_clip.write_videofile(
                abs_output,
                codec="libx264",
                audio=has_audio,
                audio_codec="aac" if has_audio else None,
                bitrate="8M",
                fps=30,
                preset="medium",
                ffmpeg_params=["-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p"],
                #ffmpeg_params=["-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
                logger="bar",
                #logger=None,
            )
        return abs_output
    except Exception as e:
        print(f"Error creating preview: {e}")
        traceback.print_exc()
        return None


def apply_ken_burns(image_path, duration=3, target_size=(1080, 1920)):
    # .resized(target_size) forces the image to cover the screen
    return (ImageClip(image_path)
            .with_duration(duration)
            .resized(target_size) # This hides the background video
            .resized(lambda t: 1 + 0.04 * t))


def assemble_final_video(original_video, plan, broll_images, transcript_data=None, output_path='output/final_short.mp4', render_mode='tiktok'):
    # 1. Load the main clip and keep it open until after rendering.
    video = VideoFileClip(original_video)
    try:
        start = float(plan.get('start', 0.0))
        end = float(plan.get('end', 0.0))
        if start < 0:
            start = 0.0
        if end < 0:
            end = 0.0
        if start >= video.duration:
            raise ValueError(f"Invalid render range: start {start} >= video duration {video.duration}")
        end = safe_clip_end(end, video.duration)
        if end <= start:
            raise ValueError(f"Invalid render range after clamping: end {end} <= start {start}")

        main_clip = video.subclipped(start, end)

        if render_mode == 'tiktok':
            w, h = main_clip.size
            target_ratio = 9 / 16
            new_w = h * target_ratio
            main_clip = main_clip.cropped(x_center=get_face_x_center(original_video) * w, width=new_w)
            # Force true HD vertical output for TikTok/Reels/Shorts.
            main_clip = resize_clip(main_clip, width=1080, height=1920)
        else:
            # Spotify clip requirement: at least 768x1024 pixels.
            if main_clip.size[0] < 768:
                main_clip = resize_clip(main_clip, width=768)
            if main_clip.size[1] < 1024:
                main_clip = resize_clip(main_clip, height=1024)

        elements = [main_clip]

        # 2. Add B-Roll with Ken Burns only in TikTok mode
        if render_mode == 'tiktok':
            for i, img_path in enumerate(broll_images):
                if i < len(plan.get('broll', [])):
                    start_time = plan['broll'][i]['time'] - plan['start']
                    if start_time < 0:
                        start_time = 0

                    duration = plan['broll'][i].get('duration', 3.0)
                    broll_clip = (apply_ken_burns(img_path, duration=duration, target_size=main_clip.size)
                                  .with_start(start_time)
                                  .with_position("center"))
                    elements.append(broll_clip)

        # 3. Add Captions from the edited transcript only in TikTok mode
        if render_mode == 'tiktok' and transcript_data:
            all_words = flatten_transcript_words(transcript_data, float(plan['start']))
            valid_words = []
            for word in all_words:
                if word["end"] <= 0 or word["start"] >= main_clip.duration:
                    continue
                word["end"] = min(main_clip.duration, word["end"])
                if word["end"] > word["start"]:
                    valid_words.append(word)
            
            dynamic_font_size = int(main_clip.size[1] * 0.0365)
            
            for group_start_idx in range(0, len(valid_words), 4):
                word_group = valid_words[group_start_idx : group_start_idx + 4]
                if not word_group:
                    continue
                elements.extend(
                    create_karaoke_word_group_clips(
                        word_group,
                        main_clip.size,
                        r"C:\Windows\Fonts\arialbd.ttf",
                        dynamic_font_size,
                        "#FFFFFF",
                        "#00FF00",
                        stroke_width=4,
                        # position_y=0.55, # old one
                        position_y=0.68, # new test
                    )
                )

        #final = CompositeVideoClip(elements)
        
        final_clip = CompositeVideoClip(elements, size=main_clip.size).with_duration(main_clip.duration)
        has_audio = main_clip.audio is not None and getattr(main_clip.audio, 'duration', 0) > 0
       
        #new changes
        clip_audio = final_clip.audio.subclipped(0, main_clip.duration)
        
        if not has_audio:
            raise ValueError("Spotify clips require audio in the final output.")
            
        #new Changes
        if hasattr(final, 'set_audio'):
            final = final.set_audio(clip_audio)
        elif hasattr(final, 'with_audio'):
            final = final.with_audio(clip_audio)
        #if hasattr(final, 'set_audio'):
        #    final = final.set_audio(main_clip.audio)
        #elif hasattr(final, 'with_audio'):
        #    final = final.with_audio(main_clip.audio)
        else:
            final.audio = main_clip.audio
        final.write_videofile(
            output_path,
            codec="libx264",
            audio=has_audio,
            audio_codec="aac",
            bitrate="8M",
            fps=30,
            preset="medium",
            ffmpeg_params=["-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p"],
        )
        return output_path
    finally:
        video.close()



