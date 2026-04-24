import yt_dlp
import os

def download_yt(url, output_filename=None):
    """
    Downloads a YouTube video and saves it as an MP4.
    """
    ydl_opts = {
        # 'bestvideo+bestaudio' can be tricky, so we ask for a combined mp4
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        # 'outtmpl': f'{output_filename}.%(ext)s',
        # This ensures it merges into a single file correctly
        'merge_output_format': 'mp4',
    }

    # If no filename provided, extract the title first
    if output_filename is None:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            output_filename = info.get('title', 'video')
        # Sanitize filename (remove invalid characters)
        output_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '-', '_')).strip()

    ydl_opts['outtmpl'] = f'{output_filename}.%(ext)s'


    print(f"--- Downloading: {url} ---")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Return the final filename (yt-dlp might add .mp4 automatically)
    return f"{output_filename}.mp4"

if __name__ == "__main__":
    # Test with a short video!
    video_url = input("Enter YouTube URL: ")
    downloaded_file = download_yt(video_url)
    print(f"Finished! File saved as: {downloaded_file}")