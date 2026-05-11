import cv2
import numpy as np
import matplotlib.pyplot as plt


# Classify if the shape is point set or contours
def classify_component(component_img):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(component_img)

    areas = stats[1:, cv2.CC_STAT_AREA]

    small = (areas < 60).sum()
    large = (areas >= 60).sum()

    if small > 10 and large < 3:
        return "contour"
    return "points"


def detect_hybrid_shapes(binary_img, min_area=120, pad=8):
    """
    {
        'box': (x,y,w,h),
        'type': 'points' or 'contour',
        'crop': image
    }
    """

    h, w = binary_img.shape[:2]
    components = []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)

    for i in range(1, num_labels): 
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crop = binary_img[y1:y2, x1:x2]

        shape_type = classify_component(crop)

        components.append({
            "box": (x1, y1, x2 - x1, y2 - y1),
            "type": shape_type,
            "crop": crop
        })

    components = sorted(components, key=lambda c: (c['box'][1], c['box'][0]))

    return components


IMAGE_PATH = r"F:\Github\bender-gestalt-analysis\data\processed_drawings\page-0008.jpg"

img = cv2.imread(IMAGE_PATH,0)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

shapes = detect_hybrid_shapes(img)

print(f"Detected shapes: {len(shapes)}")

for i, comp in enumerate(shapes):
    x, y, w, h = comp["box"]
    shape_type = comp["type"]

    cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        color,
        f"{i+1}: {shape_type}",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(color_rgb)
plt.title("Detected Shapes")
plt.axis("off")
plt.show()