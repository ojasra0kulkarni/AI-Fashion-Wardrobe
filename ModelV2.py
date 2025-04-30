import os

os.environ["ROBOFLOW_API_KEY"] = "haha"

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
from roboflow import Roboflow
import logging
import sys
from glob import glob
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FashionDetector:
    def __init__(self, roboflow_api_key="1F4zJl7SQOKZUp41xzWt"):
        self.roboflow_api_key = roboflow_api_key
        self.yolo_model = None
        self.roboflow_model = None
        self.clothing_classes = ['person', 'handbag', 'tie', 'suitcase', 'backpack']

    def load_models(self):
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLOv8 model loaded")
        except Exception as e:
            logger.error(f"YOLOv8 model loading failed: {e}")
            sys.exit(1)

        try:
            rf = Roboflow(api_key=self.roboflow_api_key)
            project = rf.workspace().project("clothing-detection-p8vmn")
            self.roboflow_model = project.version(2).model
            logger.info("Roboflow model loaded")
        except Exception as e:
            logger.error(f"Roboflow model loading failed: {e}")
            sys.exit(1)

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Invalid image: {image_path}")
            return None, None
        if img.shape[2] == 4:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr_img = img
        return img, bgr_img

    def get_dominant_color(self, img, boxes=None):
        try:
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                mask = alpha > 0
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                mask = np.ones(img.shape[:2], dtype=bool)
                rgb_img = img

            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

            if boxes:
                box_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                valid_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                        area = (x2 - x1) * (y2 - y1)
                        if area > 100:
                            box_mask[y1:y2, x1:x2] = 255
                            valid_boxes.append(box)
                        else:
                            logger.debug(f"Skipping small box: {box} (area: {area})")
                    else:
                        logger.debug(f"Invalid box: {box}")

                final_mask = cv2.bitwise_and(box_mask, mask.astype(np.uint8))
                pixels_hsv = hsv_img[final_mask > 0]
                pixels_rgb = rgb_img[final_mask > 0]
            else:
                logger.debug("No bounding boxes provided, using non-transparent pixels")
                pixels_hsv = hsv_img[mask]
                pixels_rgb = rgb_img[mask]

            if len(pixels_hsv) < 50:
                logger.warning(f"Too few pixels ({len(pixels_hsv)}), cannot determine color")
                return None

            hues = pixels_hsv[:, 0]
            saturations = pixels_hsv[:, 1]
            values = pixels_hsv[:, 2]

            valid_pixels = (saturations > 10) & (values > 10) & (values < 245)
            if not np.any(valid_pixels):
                logger.debug("No valid colored pixels, using all pixels")
                valid_pixels = np.ones_like(valid_pixels, dtype=bool)

            valid_hues = hues[valid_pixels]
            valid_rgb = pixels_rgb[valid_pixels]

            if len(valid_hues) < 20:
                logger.warning(f"Too few valid colored pixels ({len(valid_hues)}), using all pixels")
                valid_hues = hues
                valid_rgb = pixels_rgb
                if len(valid_hues) < 20:
                    logger.warning(f"Still too few pixels ({len(valid_hues)}), cannot determine color")
                    return None

            hist, bins = np.histogram(valid_hues, bins=180, range=(0, 180))
            hist = hist.astype(float)

            hist = cv2.GaussianBlur(hist.reshape(-1, 1), (9, 1), 2).flatten()

            peak_indices = np.argsort(hist)[::-1]
            dominant_hues = []
            for idx in peak_indices[:2]:
                if hist[idx] > 0.05 * hist.sum():
                    dominant_hues.append(idx)

            if not dominant_hues:
                logger.warning("No dominant hues found")
                return None

            dominant_colors_bgr = []
            for hue in dominant_hues:
                hue_mask = (valid_hues >= hue - 5) & (valid_hues <= hue + 5)
                if np.sum(hue_mask) > 10:
                    mean_rgb = np.mean(valid_rgb[hue_mask], axis=0).astype(np.uint8)
                    mean_rgb = mean_rgb.reshape(1, 1, 3)
                    mean_bgr = cv2.cvtColor(mean_rgb, cv2.COLOR_RGB2BGR)[0][0]
                    dominant_colors_bgr.append(mean_bgr)
                else:
                    logger.debug(f"Skipping hue {hue} with too few pixels")

            logger.debug(f"Detected {len(dominant_colors_bgr)} colors for {len(valid_hues)} pixels")
            return dominant_colors_bgr if dominant_colors_bgr else None
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return None

    def name_color(self, rgb):
        try:
            r, g, b = rgb
            rgb_color = sRGBColor(r / 255, g / 255, b / 255)
            hsl_color = convert_color(rgb_color, HSLColor)
            h, s, l = hsl_color.hsl_h, hsl_color.hsl_s, hsl_color.hsl_l

            if s < 0.15:
                if l > 0.9: return "white"
                if l < 0.1: return "black"
                if l > 0.7: return "light gray"
                if l < 0.3: return "dark gray"
                return "gray"

            color_ranges = [
                (345, 360, "red"), (0, 15, "red"),
                (15, 30, "red-orange"),
                (30, 45, "orange"),
                (45, 60, "yellow-orange"),
                (60, 75, "yellow"),
                (75, 105, "lime"),
                (105, 135, "yellow-green"),
                (135, 165, "green"),
                (165, 195, "cyan"),
                (195, 225, "blue"),
                (225, 255, "blue-purple"),
                (255, 285, "purple"),
                (285, 315, "magenta"),
                (315, 345, "pink")
            ]

            base_color = "unknown"
            for start, end, color in color_ranges:
                if start <= h < end or (start == 345 and h >= 345):
                    base_color = color
                    break

            saturation_modifier = ""
            if s > 0.8:
                saturation_modifier = "vivid "
            elif s < 0.3:
                saturation_modifier = "muted "

            lightness_modifier = ""
            if l > 0.8:
                lightness_modifier = "light "
            elif l < 0.2:
                lightness_modifier = "dark "

            return f"{lightness_modifier}{saturation_modifier}{base_color}".strip()
        except Exception as e:
            logger.warning(f"Color naming failed: {e}")
            return "unknown"

    def detect_clothing(self, img, image_path):
        detected_items, boxes = set(), []

        try:
            for r in self.yolo_model(img):
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    name = self.yolo_model.names[cls_id]
                    if name in self.clothing_classes:
                        detected_items.add(name)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")

        try:
            result = self.roboflow_model.predict(image_path, confidence=40, overlap=30).json()
            for pred in result.get("predictions", []):
                cls_name = pred.get("class", "").lower()
                if cls_name:
                    detected_items.add(cls_name)
                    x = pred["x"]
                    y = pred["y"]
                    w = pred["width"]
                    h = pred["height"]
                    boxes.append([int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)])
        except Exception as e:
            logger.warning(f"Roboflow detection failed: {e}")

        return sorted(detected_items), boxes

    def process_images(self, image_folder, output_csv="cropped_clothing_analysis.csv"):
        # Collect PNG image paths
        image_paths = glob(os.path.join(image_folder, "*.png"))

        if not image_paths:
            logger.error("No images found!")
            sys.exit(1)

        # Sort images by numeric suffix (e.g., cropped_image1.png, cropped_image2.png)
        def get_image_number(filename):
            match = re.search(r'cropped_image(\d+)\.png', os.path.basename(filename))
            return int(match.group(1)) if match else float('inf')

        image_paths = sorted(image_paths, key=get_image_number)

        results = []
        for path in image_paths:
            name = os.path.basename(path)
            logger.info(f"Processing {name}")
            img_rgba, img_bgr = self.load_image(path)

            if img_rgba is None or img_bgr is None:
                results.append(
                    {"Image": name, "Clothing Types": "Error", "Primary Color": "Error", "Secondary Color": "Error"})
                continue

            items, boxes = self.detect_clothing(img_bgr, path)
            colors = self.get_dominant_color(img_rgba, boxes)
            primary = self.name_color(colors[0]) if colors else "Error"
            secondary = self.name_color(colors[1]) if colors and len(colors) > 1 else "None"

            results.append({
                "Image": name,
                "Clothing Types": ", ".join(items).capitalize() if items else "None",
                "Primary Color": primary,
                "Secondary Color": secondary
            })

        pd.DataFrame(results).to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")

    def check_csv_for_none(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            none_entries = df[
                df['Clothing Types'].str.lower().isin(['none']) | df['Clothing Types'].str.lower().str.contains(
                    'not sure', na=False)]
            if not none_entries.empty:
                logger.warning("Found images with 'None' or 'Not sure' clothing types:")
                print("\nThe following clothes were not detected:")
                for _, row in none_entries.iterrows():
                    print(f"- {row['Image']}")
                return none_entries['Image'].tolist()
            else:
                logger.info("All images have identified clothing types.")
                print("\nAll clothes were detected successfully.")
                return []
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return []


def main():
    folder = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\cropped images"
    output = "cropped_clothing_analysis.csv"
    try:
        fd = FashionDetector()
        fd.load_models()
        fd.process_images(folder, output)
        fd.check_csv_for_none(output)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
