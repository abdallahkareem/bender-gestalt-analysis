import cv2, os
import numpy as np
from skimage.morphology import skeletonize


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # using gaussian filter for keep important information
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # to increase the contrast in image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Segmentation using Threshold (Binarize)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    skeleton = skeletonize(cleaned > 0)
    skeleton = (skeleton * 255).astype(np.uint8)

    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "binary": binary,
        "cleaned": cleaned,
        "skeleton": skeleton
    }


input_dir = "data/raw_drawings"
output_dir = "data/processed_drawings"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_dir, filename)

        results = preprocess_image(path)

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, results["skeleton"])

        print(f"Processed: {filename}")