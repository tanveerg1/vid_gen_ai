from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, TextClip
import os
import cv2
import numpy as np
import traceback

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
                
                print(f"Face detected: x_center={x_center:.3f}")
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


    """Split text into chunks of max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def flatten_transcript_words(transcript_data, clip_start=0.0):
    """Convert transcript entries into word-level timestamps."""
    words = []
    print(f"transcript_data: {transcript_data}")
    for entry in transcript_data:
        entry_text = str(entry.get("text", "")).strip()
        if not entry_text:
            continue 

        entry_start = float(entry.get("start", 0.0)) - float(clip_start)
        entry_end = float(entry.get("end", entry_start)) - float(clip_start)
        entry_duration = max(0.0, entry_end - entry_start)

        if entry.get("words"):
            for word in entry["words"]:
                word_text = str(word.get("text", "")).strip()
                if not word_text:
                    continue
                word_start = float(word.get("start", entry_start)) - float(clip_start)
                word_end = float(word.get("end", word_start)) - float(clip_start)
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
    stroke_width=4,
    position_y=0.78,
):
    clips = []
    group_text = " ".join(word["word"] for word in word_group)
    
    group_start = word_group[0]["start"]
    group_end = max(word["end"] for word in word_group)
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

    # Position logic
    vertical_position = int(clip_size[1] - safe_height - 250)
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



def create_fast_preview(input_path, start, end, transcript_data, output_path="output/preview_highlight.mp4"):
    """Creates a preview using MoviePy v2.0+ syntax."""
    if not os.path.exists("output"):
        os.makedirs("output")
    # print(f"Creating preview for segment {start:.2f}s to {end:.2f}s from {input_path}")
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)
    
    try:
        with VideoFileClip(abs_input) as video:
            new_clip = video.subclipped(float(start), float(end))
            
            # Crop to vertical (9:16) like the final video
            w, h = new_clip.size
            target_ratio = 9/16
            new_w = h * target_ratio
            new_clip = new_clip.cropped(x_center=get_face_x_center(abs_input) * w, width=new_w)
            
            elements = [new_clip]
            # print(f"Clip duration: {new_clip.duration:.2f}s, size: {new_clip.size}")
            all_words = flatten_transcript_words(transcript_data, float(start))
            # print(f"Total words in transcript: {len(all_words)}")
            valid_words = []
            for word in all_words:
                if word["end"] <= 0 or word["start"] >= new_clip.duration:
                    continue
                word["start"] = max(0, word["start"])
                word["end"] = min(new_clip.duration, word["end"])
                if word["end"] > word["start"]:
                    valid_words.append(word)

            for group_start_idx in range(0, len(valid_words), 4):
                word_group = valid_words[group_start_idx : group_start_idx + 4]
                if not word_group:
                    continue
                elements.extend(
                    create_karaoke_word_group_clips(
                        word_group,
                        new_clip.size,
                        r"C:\Windows\Fonts\arialbd.ttf",
                        35,
                        "#FFFFFF",
                        "#00FF00",
                        stroke_width=4,
                        position_y=0.80,
                    )
                ) 
            
            final_clip = CompositeVideoClip(elements)
            final_clip.write_videofile(
                abs_output, 
                codec="libx264", 
                audio_codec="aac", 
                fps=24, 
                preset="ultrafast",
                logger=None 
            )
        return abs_output
    except Exception as e:
        # This will now catch and print the specific error if it fails again
        print(f"Error creating preview: {e}")
        traceback.print_exc()
        return None


def apply_ken_burns(image_path, duration=3, target_size=(1080, 1920)):
    # .resized(target_size) forces the image to cover the screen
    return (ImageClip(image_path)
            .with_duration(duration)
            .resized(target_size) # This hides the background video
            .resized(lambda t: 1 + 0.04 * t))


def assemble_final_video(original_video, plan, broll_images, transcript_data=None, output_path='output/final_short.mp4'):
    # 1. Load and Crop the main clip to Vertical (9:16)
    main_clip = VideoFileClip(original_video).subclipped(plan['start'], plan['end'])
    w, h = main_clip.size
    target_ratio = 9/16
    
    new_w = h * target_ratio
    main_clip = main_clip.cropped(x_center=get_face_x_center(original_video) * w, width=new_w)
    
    elements = [main_clip]
    
    # 2. Add B-Roll with Ken Burns
    for i, img_path in enumerate(broll_images):
        if i < len(plan['broll']):
            start_time = plan['broll'][i]['time'] - plan['start']
            if start_time < 0: start_time = 0
            
            duration = plan['broll'][i].get('duration', 3.0)
            broll_clip = (apply_ken_burns(img_path, duration=duration, target_size=main_clip.size)
                          .with_start(start_time)
                          .with_position("center"))
            elements.append(broll_clip)
             
    # 3. Add Captions from the edited transcript
    if transcript_data:
        all_words = flatten_transcript_words(transcript_data, float(plan['start']))
        valid_words = []
        for word in all_words:
            if word["end"] <= 0 or word["start"] >= main_clip.duration:
                continue
            word["end"] = min(main_clip.duration, word["end"])
            if word["end"] > word["start"]:
                valid_words.append(word)

        for group_start_idx in range(0, len(valid_words), 4):
            word_group = valid_words[group_start_idx : group_start_idx + 4]
            if not word_group:
                continue
            elements.extend(
                create_karaoke_word_group_clips(
                    word_group,
                    main_clip.size,
                    r"C:\Windows\Fonts\arialbd.ttf",
                    35,
                    "#FFFFFF",
                    "#00FF00",
                    stroke_width=4,
                    position_y=0.55,
                )
            )       
        
    final = CompositeVideoClip(elements)
    
    output_path = "output/final_short.mp4"
    # Using ultrafast preset here too for testing, change to 'medium' for final quality
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
    return output_path



