from moviepy.editor import ImageClip, CompositeVideoClip

def create_moving_broll(image_path, duration=3.0):
    # Load image and set duration
    clip = ImageClip(image_path).set_duration(duration)
    
    # Apply a slow zoom-in effect (from 100% to 110% size)
    # This makes the static AI image feel like a real camera shot
    moving_clip = clip.resize(lambda t: 1 + 0.03 * t) 
    
    # Set the clip to appear in the center of a 9:16 vertical frame
    return moving_clip.set_position(("center", "center"))

def overlay_broll(main_video, broll_clips_with_times):
    # This function layers the AI images on top of your video
    clips = [main_video]
    for broll, start_time in broll_clips_with_times:
        clips.append(broll.set_start(start_time))
    
    return CompositeVideoClip(clips)