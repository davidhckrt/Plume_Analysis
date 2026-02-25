import os
import subprocess
import cv2
import numpy as np

def create_directories(base_path):
    """Create directories for original, enhanced, and edge frames."""
    dirs = {
        'original': os.path.join(base_path, 'original_frames'),
        'enhanced': os.path.join(base_path, 'enhanced_frames'),
        'edges': os.path.join(base_path, 'edge_frames')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def extract_frames(video_path, output_dir):
    """Extract frames using FFmpeg."""
    cmd = ['ffmpeg', '-i', video_path, 
           '-vf', 'fps=3',  # 3 frames per second
           os.path.join(output_dir, 'frame_%04d.png')]
    subprocess.run(cmd, check=True)

def enhance_contrast(img):
    """Enhance contrast of the image while preserving color."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def process_frames(input_dir, original_dir, enhanced_dir, edge_dir):
    """Process frames with contrast enhancement and edge detection."""
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Enhance contrast while keeping color
            enhanced_img = enhance_contrast(img)
            
            # Edge detection
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            edge_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Add frame number and timestamp
            frame_num = filename.split('_')[1].split('.')[0]
            timestamp_min = f"{int(float(frame_num)/5/60):02d}:{int(float(frame_num)/5%60):02d}"
            
            # Add white box with text
            cv2.rectangle(edge_img, (img.shape[1]-210, 10), (img.shape[1]-10, 50), (255,255,255), -1)
            cv2.putText(edge_img, f"Frame: {frame_num} | {timestamp_min}", 
                        (img.shape[1]-200, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            # Save images
            cv2.imwrite(os.path.join(original_dir, filename), img)
            cv2.imwrite(os.path.join(enhanced_dir, filename), enhanced_img)
            cv2.imwrite(os.path.join(edge_dir, filename), edge_img)

def main():
    """Main processing function."""
    # Get user input
    video_path = input("Enter the path to the video file: ").strip()
    output_base_path = input("Enter the base output folder: ").strip()
    
    # Create directories
    dirs = create_directories(output_base_path)
    
    # Temporary directory for initial extraction
    temp_dir = os.path.join(output_base_path, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract frames
    extract_frames(video_path, temp_dir)
    
    # Process frames
    process_frames(temp_dir, 
                   dirs['original'], 
                   dirs['enhanced'], 
                   dirs['edges'])
    
    # Clean up temporary directory
    subprocess.run(['rm', '-rf', temp_dir])
    print(f"Processing complete. Frames saved to: {output_base_path}")

if __name__ == "__main__":
    main()
