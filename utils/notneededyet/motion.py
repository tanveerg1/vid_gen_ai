from moviepy.editor import ImageClip

def create_animated_broll(image_path, duration=4.0):
    # Load the image as a clip
    clip = ImageClip(image_path).set_duration(duration)
    
    # Simple zoom-in effect: 100% to 115% size over the duration
    # This creates a "cinematic" feel
    animated = clip.resize(lambda t: 1 + 0.04 * t)
    
    # Center the clip for vertical 9:16
    return animated.set_position(("center", "center"))