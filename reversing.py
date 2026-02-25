import os
import subprocess

def reverse_video(input_path):
    """
    Reverse a video using FFmpeg and save it in the same folder as the input.
    
    Parameters:
        input_path (str): Path to the input video file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist at {input_path}")
        return

    # Create output file name in the same folder
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_reversed{ext}"

    # Reverse the video
    command = [
        "ffmpeg", "-i", input_path,
        "-vf", "reverse", "-af", "areverse",
        output_path
    ]

    print(f"Reversing video: {input_path}")
    try:
        subprocess.run(command, check=True)
        print(f"Reversed video saved as: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while reversing video: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # Get video file path from user
    video_path = input("Enter the path to the video file: ").strip()

    # Reverse the video
    reverse_video(video_path)

if __name__ == "__main__":
    main()
