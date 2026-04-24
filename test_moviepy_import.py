import moviepy
print("MoviePy submodules:", moviepy.__path__)

# Try different import paths
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    print("✓ Found: moviepy.video.io.VideoFileClip")
except Exception as e:
    print(f"✗ moviepy.video.io.VideoFileClip: {e}")

try:
    from moviepy.video.VideoFileClip import VideoFileClip
    print("✓ Found: moviepy.video.VideoFileClip")
except Exception as e:
    print(f"✗ moviepy.video.VideoFileClip: {e}")

try:
    from moviepy.video.io import VideoFileClip
    print("✓ Found: moviepy.video.io (module)")
except Exception as e:
    print(f"✗ moviepy.video.io: {e}")

import pkgutil
print("\nMoviespy modules:")
for importer, modname, ispkg in pkgutil.iter_modules(moviepy.__path__):
    print(f"  {modname}")
