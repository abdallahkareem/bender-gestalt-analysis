import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

# =========================
# Configuration
# =========================
TEMPLATE_DIR = r"D:\assignments\cognitive\project\bender-gestalt-analysis\data\templates"
DETECTED_DIR = r"D:\assignments\cognitive\project\bender-gestalt-analysis\data\Detected"
OUTPUT_CSV = "bender_analysis_report.csv"

# Thresholds
HU_THRESHOLD = 14.5
HAUSDORFF_THRESHOLD = 122.5
AREA_RATIO_THRESHOLD = 180


# =========================
# Image Processing
# =========================
def load_binary_image(path, size=(256, 256)):
    """
    Load image → grayscale → resize → binary.
    No cropping (assumed already preprocessed).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(img, size)

    _, binary = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


# =========================
# Feature Extraction
# =========================
def compute_hu_moments(binary):
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def hu_distance(img1, img2):
    return np.linalg.norm(
        compute_hu_moments(img1) - compute_hu_moments(img2)
    )


def extract_points(binary):
    pts = np.column_stack(np.where(binary > 0))
    return pts if len(pts) > 0 else np.array([[0, 0]])


def hausdorff_distance(img1, img2):
    p1 = extract_points(img1)
    p2 = extract_points(img2)

    return max(
        directed_hausdorff(p1, p2)[0],
        directed_hausdorff(p2, p1)[0]
    )


def area(binary):
    return cv2.countNonZero(binary)


def area_ratio(template_img, student_img):
    t = area(template_img)
    s = area(student_img)
    return 0 if t == 0 else s / t


def count_components(binary):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)
    return num_labels - 1


# =========================
# Error Detection
# =========================
def detect_errors(hu_dist, haus_dist, ar, t_comp, s_comp):
    errors = []

    if hu_dist > HU_THRESHOLD and haus_dist > HAUSDORFF_THRESHOLD:
        errors.append("Distortion")

    if ar < AREA_RATIO_THRESHOLD:
        errors.append("Omission")

    if s_comp > t_comp:
        errors.append("Integration Failure")

    if ar > AREA_RATIO_THRESHOLD + 10:
        errors.append("Perseveration")

    if not errors:
        errors.append("No Significant Errors")

    return ", ".join(errors)


# =========================
# Compare Shapes
# =========================
def compare_shape(template_img, student_img):

    hu_dist = hu_distance(template_img, student_img)
    haus_dist = hausdorff_distance(template_img, student_img)
    ar = area_ratio(template_img, student_img)

    t_comp = count_components(template_img)
    s_comp = count_components(student_img)

    errors = detect_errors(
        hu_dist, haus_dist, ar, t_comp, s_comp
    )

    return {
        "hu_distance": round(hu_dist, 4),
        "hausdorff_distance": round(haus_dist, 4),
        "area_ratio": round(ar, 4),
        "template_components": t_comp,
        "student_components": s_comp,
        "errors": errors,
    }


# =========================
# Template Matching (Hu-based)
# =========================
def find_best_template(student_img):
    best_hu = float("inf")
    best_template = None

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for t_name in os.listdir(TEMPLATE_DIR):
        if not t_name.lower().endswith(valid_ext):
            continue

        t_path = os.path.join(TEMPLATE_DIR, t_name)

        try:
            t_img = load_binary_image(t_path)
            hu_d = hu_distance(t_img, student_img)

            if hu_d < best_hu:
                best_hu = hu_d
                best_template = (t_path, t_name)

        except Exception as e:
            print(f"Template error {t_name}: {e}")

    if best_template is None:
        raise ValueError("No valid template found")

    return best_template[0], best_template[1], best_hu


# =========================
# Main Pipeline
# =========================
def analyze_all_students():
    results = []

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for student_name in sorted(os.listdir(DETECTED_DIR)):
        student_folder = os.path.join(DETECTED_DIR, student_name)

        if not os.path.isdir(student_folder):
            continue

        print(f"\nAnalyzing {student_name}...")

        for shape_name in sorted(os.listdir(student_folder)):

            if not shape_name.lower().endswith(valid_ext):
                continue

            student_path = os.path.join(student_folder, shape_name)

            try:
                student_img = load_binary_image(student_path)

                # Step 1: find best template
                t_path, t_name, hu_init = find_best_template(student_img)

                print(f"  {shape_name} -> {t_name} (Hu={hu_init:.4f})")

                # Step 2: full comparison
                metrics = compare_shape(
                    load_binary_image(t_path),
                    student_img
                )

                results.append({
                    "student": student_name,
                    "shape": shape_name,
                    "matched_template": t_name,
                    **metrics
                })

            except Exception as e:
                print(f"Error {student_name}/{shape_name}: {e}")

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by=["student", "matched_template"])

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    analyze_all_students()