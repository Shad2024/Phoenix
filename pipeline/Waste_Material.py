from ultralytics import YOLO
import cv2
import os
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict


model_path = r"C:\Users\bolaky\PycharmProjects\Rebuild\Models\DetectionModel\segment\train\weights\best.pt"
model = YOLO(model_path)

captured_folder = Path(r"C:\Users\bolaky\PycharmProjects\Rebuild\outputs\captured_images")
output_folder = Path(r"C:\Users\bolaky\PycharmProjects\Rebuild\outputs\Material_results")
output_folder.mkdir(parents=True, exist_ok=True)

def save_segmentation_data(csv_path, image_name, class_name, pixel_area):
    # Create a one-row DataFrame
    data = pd.DataFrame([{
        "image_name": image_name,
        "label": class_name,
        "pixel_size": pixel_area
    }])

    # Append to CSV file
    header = not os.path.exists(csv_path)
    data.to_csv(csv_path, mode='a', index=False, header=header)


# Get latest folder of images
subfolders = [f for f in captured_folder.iterdir() if f.is_dir()]
if not subfolders:
    raise FileNotFoundError(" No subfolders found inside captured_images/")

latest_folder = max(subfolders, key=os.path.getmtime)
print(f" Latest folder selected: {latest_folder.name}")

# Get all images in that folder

image_types = ('*.jpg', '*.jpeg', '*.png')
image_paths = []
for ext in image_types:
    image_paths.extend(glob.glob(str(latest_folder / ext)))

if not image_paths:
    raise FileNotFoundError(f"❌ No images found in {latest_folder}")

print(f"🖼 Found {len(image_paths)} images. Starting segmentation...")

# PROCESS EACH IMAGE
for image_path in sorted(image_paths):
    img_name = os.path.basename(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Could not read {image_path}")
        continue

    print(f"🔍 Detecting materials in: {Path(image_path).name}")
    results = model.predict(source=image, conf=0.5, save=False)

    # Dictionary to store total area per material
    material_areas = defaultdict(float)

    # Draw detections
    for r in results:
        masks = r.masks

        if masks is not None:
            mask_data = masks.data.cpu().numpy()
            for i, mask in enumerate(mask_data):
                cls_id = int(r.boxes.cls[i])
                class_name = model.names[cls_id]

                # Create color for each class
                color = tuple(np.random.randint(0, 255, size=3).tolist())

                # Resize mask to match the image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_uint8 = (mask_resized * 255).astype(np.uint8)

                # Find contours
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)

                # Apply transparent overlay
                overlay = image.copy()
                overlay[mask_uint8 > 0] = color
                image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

                # Calculate mask area
                area = np.sum(mask_uint8 > 0)
                material_areas[class_name] += area
                save_segmentation_data("../segmentation_results.csv", img_name, class_name, area)

                # Label text
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.putText(image, class_name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            print(" No segmentation masks found for this image.")

        # -------------------------
        # ✅ 5️⃣ Calculate & Aggregate Percentages (New Clean Version)
        # -------------------------
    total_area = sum(material_areas.values())
    material_percentages = {
        mat: (area / total_area) * 100 if total_area > 0 else 0
        for mat, area in material_areas.items()
    }

    summary_path = "../outputs/segmentation_summary.csv"

    # Load previous data if available
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
    else:
        existing = pd.DataFrame(columns=["image_name", "material", "percentage"])

    # Remove old entries for this image before adding new ones
    existing = existing[existing["image_name"] != img_name]

    # Add updated rows for current image
    for material, percentage in material_percentages.items():
        new_row = {"image_name": img_name, "material": material, "percentage": percentage}
        existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)

    # Save clean aggregated summary
    existing.to_csv(summary_path, index=False)
    print(f" Updated aggregated summary for {img_name}")

    # Write material report on the image
    y0, dy = 30, 30
    for i, (mat, pct) in enumerate(material_percentages.items()):
        text = f"{mat}: {pct:.1f}%"
        cv2.putText(image, text, (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save annotated image
    save_path = output_folder / f"{Path(image_path).stem}_detected.jpg"
    cv2.imwrite(str(save_path), image)

    # Show result
    cv2.imshow("Material Detection", image)
    print(f"✅ Saved and displayed: {save_path.name}")
    cv2.waitKey(500)  # Show for 30 seconds

cv2.destroyAllWindows()
print(" All images processed successfully.")