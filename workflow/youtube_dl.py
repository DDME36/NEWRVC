import os
import yt_dlp
from urllib.parse import urlparse

def get_youtube_audio(url: str, output_dir: str) -> str:
    """
    Downloads the best audio from a YouTube URL and returns the path to the downloaded .wav file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "quiet": False,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            },
        ],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        if not result:
            raise ValueError(f"Could not extract info from URL: {url}")
        file_path = ydl.prepare_filename(result)
        
    # The postprocessor changes the extension to .wav
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    return wav_path

def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in ["youtu.be", "www.youtube.com", "youtube.com", "music.youtube.com"]
