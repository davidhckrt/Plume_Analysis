import cv2
import os
import csv
import numpy as np
from tqdm import tqdm

class PolygonAnnotator:
    def __init__(self, input_folder, output_folder, csv_file="annotations.csv"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.csv_file = csv_file
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize state variables
        self.polygon = []
        self.previous_polygon = []  # Store the previous frame's polygon
        self.frame = None
        self.clone = None
        self.show_magnifier = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.limit_y = None
        self.setting_limit = False
        
    def create_magnifying_glass(self, x, y, size=200):
        """Create a magnified view of a specific region."""
        if self.frame is None:
            return None
            
        height, width = self.frame.shape[:2]
        half_size = size // 2
        
        # Ensure the magnification region stays within image bounds
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(width, x + half_size)
        y2 = min(height, y + half_size)
        
        # Crop and resize the region
        region = self.frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
            
        magnified = cv2.resize(region, (size, size))
        
        # Create circular mask
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size//2, size//2), size//2, 255, -1)
        
        # Apply mask and add crosshair
        result = cv2.bitwise_and(magnified, magnified, mask=mask)
        center = size // 2
        cv2.line(result, (center, center-10), (center, center+10), (0, 255, 0), 1)
        cv2.line(result, (center-10, center), (center+10, center), (0, 255, 0), 1)
        cv2.circle(result, (size//2, size//2), size//2, (0, 255, 0), 2)
        
        return result
        
    def print_help(self):
        """Print help information to terminal."""
        help_text = """
Keyboard Shortcuts:
------------------
Space: Complete polygon
R: Reset current frame
Q: Skip frame
B: Go back to previous frame
G: Toggle magnifying glass
L: Set horizontal line (click to place)
Esc: Exit program

Mouse Controls:
--------------
Left click:  Add vertex
Right click: Remove last vertex

Note: Previous frame's polygon shown in gray during annotation
"""
        print(help_text)

    def draw_limit_line(self):
        """Draw the horizontal limit line if it exists."""
        if self.limit_y is not None:
            cv2.line(self.frame, (0, self.limit_y), 
                    (self.frame.shape[1], self.limit_y), (0, 0, 255), 2)  # Red line
                    
    def draw_previous_polygon(self, image):
        """Draw the previous frame's polygon in gray."""
        if len(self.previous_polygon) > 2:
            # Draw previous polygon in gray
            for i in range(len(self.previous_polygon) - 1):
                cv2.line(image, self.previous_polygon[i], self.previous_polygon[i+1], (128, 128, 128), 2)
            # Close the polygon
            cv2.line(image, self.previous_polygon[-1], self.previous_polygon[0], (128, 128, 128), 2)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.setting_limit:
                self.limit_y = y
                self.setting_limit = False
                self.draw_limit_line()  # Draw the line immediately
                print(f"Line set at y = {y}")
            else:
                self.polygon.append((x, y))
                cv2.circle(self.frame, (x, y), 3, (0, 255, 0), -1)
                if len(self.polygon) > 1:
                    cv2.line(self.frame, self.polygon[-2], self.polygon[-1], (255, 0, 0), 2)
            self.update_display()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.polygon:
                self.polygon.pop()
                self.frame = self.clone.copy()
                self.draw_limit_line()  # Redraw the limit line after clearing
                self.redraw_polygon()
                self.update_display()
                
    def update_display(self):
        """Update the display with current frame and overlays."""
        display = self.frame.copy()
        
        # Draw the previous polygon in gray
        self.draw_previous_polygon(display)
        
        # Draw limit line if it exists
        if self.limit_y is not None:
            cv2.line(display, (0, self.limit_y), 
                    (display.shape[1], self.limit_y), (0, 0, 255), 2)
        
        if self.show_magnifier:
            magnified = self.create_magnifying_glass(self.mouse_x, self.mouse_y)
            if magnified is not None:
                h, w = magnified.shape[:2]
                display[10:10+h, -h-10:-10] = magnified
                
        cv2.imshow("Polygon Annotation", display)
                
    def redraw_polygon(self):
        """Redraw the current polygon."""
        for i, point in enumerate(self.polygon):
            cv2.circle(self.frame, point, 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(self.frame, self.polygon[i-1], point, (255, 0, 0), 2)

    def draw_text_box(self, image, text, position, font_scale=0.6, thickness=1):
        """Draw text in a white box on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate box coordinates
        padding = 5  # Reduced padding
        box_x1 = position[0]
        box_y1 = position[1]
        box_x2 = box_x1 + text_width + 2 * padding
        box_y2 = box_y1 + text_height + 2 * padding
        
        # Draw white box with black border
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        
        # Draw black text
        text_x = box_x1 + padding
        text_y = box_y1 + text_height + padding - 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return image
                
    def calculate_height(self):
        """Calculate and draw height measurement."""
        if len(self.polygon) < 2:
            return 0
            
        highest = min(self.polygon, key=lambda p: p[1])
        lowest = max(self.polygon, key=lambda p: p[1])
        
        # Draw vertical line
        cv2.line(self.frame, (lowest[0], highest[1]), (lowest[0], lowest[1]), (0, 0, 255), 2)
        
        # Calculate height
        height = lowest[1] - highest[1]
        
        # Close the polygon for display
        closed_polygon = self.polygon + [self.polygon[0]]
        for i in range(len(closed_polygon) - 1):
            cv2.line(self.frame, closed_polygon[i], closed_polygon[i+1], (255, 0, 0), 2)
        
        # Draw height measurement in white box
        text = f"Height: {height} px"
        box_x = lowest[0] + 20
        box_y = (highest[1] + lowest[1]) // 2
        self.draw_text_box(self.frame, text, (box_x, box_y))
        
        # Redraw limit line if it exists
        self.draw_limit_line()
        
        return height
        
    def run(self):
        """Main annotation loop."""
        self.print_help()
        
        # Get list of image files
        image_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.png'))])
        if not image_files:
            print("No images found in input folder!")
            return
            
        # Open CSV file
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Height (px)", "Polygon Points"])
            
            current_frame_idx = 0
            
            while current_frame_idx < len(image_files):
                image_file = image_files[current_frame_idx]
                image_path = os.path.join(self.input_folder, image_file)
                
                # Load image
                self.frame = cv2.imread(image_path)
                if self.frame is None:
                    print(f"Error loading {image_file}")
                    current_frame_idx += 1
                    continue
                    
                self.clone = self.frame.copy()
                self.polygon = []
                
                # Draw the limit line if it exists
                self.draw_limit_line()
                
                # Setup window
                cv2.namedWindow("Polygon Annotation")
                cv2.setMouseCallback("Polygon Annotation", self.mouse_callback)
                
                print(f"\nAnnotating: {image_file} ({current_frame_idx + 1}/{len(image_files)})")
                
                while True:
                    self.update_display()
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27:  # Esc key
                        print("\nExiting...")
                        cv2.destroyAllWindows()
                        return
                        
                    elif key == 32:  # Space key
                        if len(self.polygon) > 2:
                            # Save the current polygon before calculating height
                            self.previous_polygon = self.polygon.copy()
                            
                            # Calculate height and save without previous polygon overlay
                            height = self.calculate_height()
                            closed_polygon = self.polygon + [self.polygon[0]]
                            writer.writerow([image_file, height, closed_polygon])
                            
                            output_path = os.path.join(self.output_folder, image_file)
                            # Save the frame without the previous polygon overlay
                            cv2.imwrite(output_path, self.frame)
                            print(f"Saved annotation for {image_file}")
                            current_frame_idx += 1
                            break
                        else:
                            print("Need at least 3 points!")
                            
                    elif key == ord('b'):  # Back to previous frame
                        if current_frame_idx > 0:
                            # Clear previous polygon when going back
                            self.previous_polygon = []
                            current_frame_idx -= 1
                            print("Going back to previous frame")
                            break
                        else:
                            print("Already at first frame")
                            
                    elif key == ord('l'):  # Set limit line
                        self.setting_limit = True
                        print("Click to set the horizontal line")
                            
                    elif key == ord('r'):  # Reset current frame only
                        self.frame = self.clone.copy()
                        self.polygon = []
                        print("Reset current frame")
                        # Redraw the limit line after reset
                        self.draw_limit_line()
                        
                    elif key == ord('q'):  # Skip
                        # Save current polygon as previous before skipping
                        if len(self.polygon) > 2:
                            self.previous_polygon = self.polygon.copy()
                        print("Skipped frame")
                        current_frame_idx += 1
                        break
                        
                    elif key == ord('g'):  # Toggle magnifying glass
                        self.show_magnifier = not self.show_magnifier
                        print(f"Magnifying glass {'enabled' if self.show_magnifier else 'disabled'}")
                        
        print(f"\nAnnotations saved to {self.csv_file}")
        cv2.destroyAllWindows()

def get_valid_path():
    """Ask user for input path and validate it."""
    while True:
        path = input("Enter the path to the folder containing frames to annotate: ").strip()
        
        # Handle if user wraps path in quotes
        path = path.strip('"\'')
        
        if os.path.exists(path):
            # Check if directory contains any images
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
            if image_files:
                return path
            else:
                print("Error: No JPG or PNG files found in the specified directory.")
        else:
            print("Error: Directory does not exist. Please enter a valid path.")

def main():
    # Get input path from user
    input_folder = get_valid_path()
    
    # Create output folder name based on input folder
    output_folder = os.path.join(os.path.dirname(input_folder), "annotated_frames")
    
    # Create CSV filename based on input folder
    base_name = os.path.basename(input_folder)
    csv_file = f"polygon_annotations_{base_name}.csv"
    
    print(f"\nAnnotation will be saved to:")
    print(f"Images: {output_folder}")
    print(f"Data: {csv_file}")
    print("\nPress Enter to start annotation, or Ctrl+C to cancel...")
    input()
    
    # Start annotation
    annotator = PolygonAnnotator(input_folder, output_folder, csv_file)
    annotator.run()

if __name__ == "__main__":
    main()
