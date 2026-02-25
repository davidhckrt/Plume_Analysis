import cv2
import numpy as np
import os

def process_frame(frame, ref_image, ref_kp, ref_des):
    """
    Given an input frame, compute its SIFT features and match them with the reference image.
    Compute the homography (from reference to frame) and invert it to warp the frame into the
    coordinate system of the reference image.
    Returns the warped frame (or None if matching fails).
    """
    # Convert the frame to grayscale and detect SIFT features
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    if des_frame is None:
        return None

    # Use FLANN-based matcher to match descriptors between reference and frame
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(ref_des, des_frame, k=2)
    except cv2.error as e:
        print("FLANN matching error:", e)
        return None

    # Apply Lowe's ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) < 10:
        return None

    # Extract point correspondences from the good matches
    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography from reference image to the frame using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # Invert the homography to map the current frame into the reference coordinate system
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    ref_h, ref_w = ref_image.shape[:2]
    # Warp the current frame to align with the reference image's coordinate system
    warped_frame = cv2.warpPerspective(
        frame, H_inv, (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped_frame

def main():
    # Prompt the user for the reference image, input folder, and output folder.
    ref_path = input("Enter the path to the reference (zoomed) image: ").strip()
    in_folder = input("Enter the path to the folder containing input frames: ").strip()
    out_folder = input("Enter the path to the folder for output frames: ").strip()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Load and process the reference image.
    ref_image = cv2.imread(ref_path)
    if ref_image is None:
        print("Error: Could not load the reference image.")
        return

    # Compute SIFT features on the reference image.
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    ref_kp, ref_des = sift.detectAndCompute(gray_ref, None)
    if ref_des is None:
        print("Error: No features detected in the reference image.")
        return

    # Get a sorted list of input frame filenames.
    frame_files = sorted([f for f in os.listdir(in_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    last_valid_frame = None

    for fname in frame_files:
        frame_path = os.path.join(in_folder, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load {fname}. Skipping.")
            continue

        stabilized = process_frame(frame, ref_image, ref_kp, ref_des)
        if stabilized is None:
            print(f"Warning: Processing failed for {fname}. Skipping.")
            continue

        # Fill black areas (resulting from the warp) with pixels from the last valid frame,
        # if available.
        if last_valid_frame is not None:
            mask = (stabilized == 0).all(axis=2)
            stabilized[mask] = last_valid_frame[mask]

        # Update last valid frame with the current stabilized frame.
        last_valid_frame = stabilized.copy()

        # Save the stabilized frame with the same name into the output folder.
        out_path = os.path.join(out_folder, fname)
        cv2.imwrite(out_path, stabilized)
        print(f"Processed {fname} -> saved to {out_path}")

        # Optionally, display the stabilized frame.
        cv2.imshow("Stabilized Frame", stabilized)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit early
            break

    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
