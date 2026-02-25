import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.tracked_features = []  # List of tracked features
        self.prev_gray = None       # Previous frame in grayscale
        self.fresh_start = True     # Flag for reset
        self.rigid_transform = np.eye(3, dtype=np.float32)  # Affine 2x3 in a 3x3 matrix

    def process_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        corners = []

        # Detect new features if less than 200 are being tracked
        if len(self.tracked_features) < 200:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=10)
            if corners is not None:
                corners = corners.reshape(-1, 2)
                print(f"Found {len(corners)} features")
                self.tracked_features.extend(corners)

        # Perform feature tracking
        if self.prev_gray is not None and self.tracked_features:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                np.array(self.tracked_features, dtype=np.float32),
                None
            )

            # Filter features based on tracking status
            valid_indices = np.where(status.flatten() == 1)[0]
            new_features = new_features[valid_indices]
            old_features = np.array(self.tracked_features, dtype=np.float32)[valid_indices]

            # Handle catastrophic error: if too few features are tracked
            if len(new_features) < 0.8 * len(self.tracked_features):
                print("Catastrophic error: Resetting tracker.")
                self.rigid_transform = np.eye(3, dtype=np.float32)
                self.tracked_features.clear()
                self.prev_gray = None
                self.fresh_start = True
                return

            # Estimate affine transform
            transform_matrix, _ = cv2.estimateAffinePartial2D(old_features, new_features, method=cv2.RANSAC)
            if transform_matrix is not None:
                transform_3x3 = np.eye(3, dtype=np.float32)
                transform_3x3[:2] = transform_matrix  # Embed 2x3 affine transform into a 3x3 matrix
                self.rigid_transform = self.rigid_transform @ transform_3x3

            # Update tracked features
            self.tracked_features = [tuple(pt) for pt in new_features]

        # Update the previous frame
        self.prev_gray = gray


def main():
    # Prompt user for video path
    video_path = input("Enter the path to the video file: ").strip()
    output_with_points = video_path.replace(".mp4", "_with_points.mp4")
    output_without_points = video_path.replace(".mp4", "_without_points.mp4")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_with_points = cv2.VideoWriter(output_with_points, fourcc, fps, (frame_width, frame_height))
    out_without_points = cv2.VideoWriter(output_without_points, fourcc, fps, (frame_width, frame_height))

    tracker = Tracker()
    last_valid_frame = None  # To store the last visible frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()  # Copy the original frame
        tracker.process_image(orig)  # Process the frame with the tracker

        # Apply the inverse of the accumulated rigid transform
        inv_transform = np.linalg.inv(tracker.rigid_transform)
        stabilized_frame = cv2.warpAffine(
            frame,
            inv_transform[:2],
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Fill black borders with the last valid frame
        if last_valid_frame is not None:
            mask = (stabilized_frame == 0).all(axis=2)  # Find black regions in the frame
            stabilized_frame[mask] = last_valid_frame[mask]  # Fill with the last valid frame

        last_valid_frame = stabilized_frame.copy()  # Update the last valid frame

        # Create a version with tracking points
        frame_with_points = stabilized_frame.copy()
        for pt in tracker.tracked_features:
            cv2.circle(frame_with_points, tuple(map(int, pt)), 2, (0, 0, 255), -1)  # Smaller red points

        # Show the stabilized frame with tracking points
        cv2.imshow("Stabilized (With Points)", frame_with_points)

        # Write both versions to their respective output files
        out_with_points.write(frame_with_points)
        out_without_points.write(stabilized_frame)

        # Break on 'ESC'
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    out_with_points.release()
    out_without_points.release()
    cv2.destroyAllWindows()
    print(f"Videos saved as:\n - With points: {output_with_points}\n - Without points: {output_without_points}")


if __name__ == "__main__":
    main()
