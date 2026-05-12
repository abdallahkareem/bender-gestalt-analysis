import cv2
import numpy as np
import os

img_folder = r"D:\hady\3nd year\Second_term\Cognitive\project\bender-gestalt-analysis\data\processed_drawings"
base_output_path = r"D:\hady\3nd year\Second_term\Cognitive\project\bender-gestalt-analysis\data\Detected"

os.makedirs(base_output_path, exist_ok=True)

for filename in os.listdir(img_folder):

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        student_id = os.path.splitext(filename)[0]
        student_folder = os.path.join(base_output_path, student_id)
        os.makedirs(student_folder, exist_ok=True)

        full_path = os.path.join(img_folder, filename)
        img = cv2.imread(full_path, 0)  # grayscale مباشرة

        if img is None:
            print(" Cannot read:", full_path)
            continue

        # optional: denoise
        img = cv2.medianBlur(img, 5)

        # dilation 
        kernel = np.ones((12, 12), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        shape_count = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 7000:

                x, y, w, h = cv2.boundingRect(cnt)

                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)

                roi = img[y1:y2, x1:x2]

                save_path = os.path.join(student_folder, f"shape_{shape_count}.jpg")
                cv2.imwrite(save_path, roi)

                shape_count += 1

print(f" Done! Output saved in: {base_output_path}")