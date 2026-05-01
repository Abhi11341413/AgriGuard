import cv2
import numpy as np
from PIL import Image

def calculate_disease_severity(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    total_image_pixels = img_array.shape[0] * img_array.shape[1]

    # --- 1. RELAXED GREEN MASK ---
    # Lowered Hue to 35 to catch the yellowish-green transition zones.
    # Lowered Saturation to 50 to catch sun glare / washed out areas on the leaf.
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- 2. DISEASE MASK (Browns/Yellows) ---
    lower_disease = np.array([10, 50, 20])
    upper_disease = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)

    healthy_pixels = cv2.countNonZero(green_mask)
    diseased_pixels = cv2.countNonZero(disease_mask)
    total_plant_pixels = healthy_pixels + diseased_pixels

    # --- THE ADJUSTED BOUNCER ---
    if total_image_pixels == 0:
        return -1.0, "Not a Leaf"

    # 1. Absolute Green Check: Dropped to 1% to allow zoomed-in diseased leaves.
    green_ratio_total = healthy_pixels / total_image_pixels
    if green_ratio_total < 0.01:
        return -1.0, "Not a Leaf"

    if total_plant_pixels == 0:
        return -1.0, "Not a Leaf"

    # 2. The Face/Desk Trap: Dropped from 10% to 2%.
    # Highly severe diseases like Late Blight can destroy 95%+ of a leaf.
    # We only want to block objects that are mathematically 98%+ brown/orange (like faces).
    if healthy_pixels < (0.02 * total_plant_pixels):
         return -1.0, "Not a Leaf"

    # --- Proceed with normal severity math ---
    severity_ratio = (diseased_pixels / total_plant_pixels) * 100
    severity_percentage = round(severity_ratio, 2)

    if severity_percentage < 15:
        category = "Low"
    elif severity_percentage < 40:
        category = "Moderate"
    else:
        category = "High"

    return severity_percentage, category