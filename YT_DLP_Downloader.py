import os
import sys
import subprocess
import json

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("Installing yt-dlp...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    from yt_dlp import YoutubeDL


def download_youtube_video(video_url, output_file):
    """
    Download the YouTube video with the best quality and save metadata.

    Parameters:
        video_url (str): The URL of the YouTube video.
        output_file (str): Output file path.
    """
    # Prepare output template (without extension)
    output_template = os.path.splitext(output_file)[0]

    options = {
        "format": "bv+ba/b",  # Best video and audio quality
        "merge_output_format": "mp4",  # Output as MP4
        "outtmpl": output_template + ".%(ext)s",  # Ensure correct extension
        "writesubtitles": False,  # If you also want subtitles, set to True
        "writeinfojson": True,  # Save metadata to a .info.json file
        "postprocessors": [{
            "key": "FFmpegMetadata",  # Embed metadata into the video file itself
        }],
    }

    with YoutubeDL(options) as ydl:
        ydl.download([video_url])


def main():
    # Get inputs
    video_url = input("Enter YouTube video URL: ").strip()
    output_file = input("Enter output file name (e.g., video.mp4): ").strip()

    # Ensure output file has .mp4 extension
    if not output_file.endswith(".mp4"):
        output_file += ".mp4"

    try:
        print("Starting download...")
        download_youtube_video(video_url, output_file)
        print(f"Download complete. Saved to {output_file}")
        
        # Inform user about metadata file
        metadata_file = os.path.splitext(output_file)[0] + ".info.json"
        if os.path.exists(metadata_file):
            print(f"Metadata saved to {metadata_file}")
        else:
            print("Metadata file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
