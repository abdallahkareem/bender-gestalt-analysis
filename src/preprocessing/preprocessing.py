import cv2, os
import numpy as np
from skimage.morphology import skeletonize


def preprocess_image(image_path, return_skeleton=False):
    """
    Robust preprocessing pipeline for hand-drawn shapes.
    
    Returns:
        dict: contains processed images
    """

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoising (preserve edges)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Contrast enhancement (very important for pencil drawings)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Adaptive Threshold (best for scanned drawings)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # 6. Morphological Cleaning (VERY IMPORTANT FIX)
    kernel = np.ones((3, 3), np.uint8)

    # close gaps first (connect broken strokes)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # remove small noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # optional: strengthen shapes
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    '''
    # 7. Skeleton (optional ONLY for analysis, NOT detection)
    skeleton = None
    if return_skeleton:
        sk = skeletonize(cleaned > 0)
        skeleton = (sk * 255).astype(np.uint8)
    '''
    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "binary": binary,
        "cleaned": cleaned,
        # "skeleton": skeleton
    }


input_dir = "data/rraw_drawings"
output_dir = "data/processed_drawings"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_dir, filename)

        results = preprocess_image(path)

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, results["cleaned"])

        print(f"Processed: {filename}")