from datetime import datetime
import os
import time
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import re
import json 

def sorted_nicely(file_list):
    """Sort filenames like image_1, image_2, image_10 in natural numeric order."""
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

def load_model(checkpoint_path, device):
    print("🔹 Loading xView2 model...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    # Load weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(" Model loaded successfully.")
    return model


def preprocess_frame(frame):
    frame = cv2.resize(frame, (512, 512))
    img = frame.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def analyze_damage(model,frame,device):
    img = preprocess_frame(frame).to(device)
    with torch.no_grad():
        pred = model(img)
        mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy()

    labels = ["no damage", "minor damage", "major damage", "destroyed"]
    total_pixels = mask.size

    mask = np.array(mask, dtype=np.uint8)
    counts = {label: int((mask == i).sum()) for i, label in enumerate(labels)}
    percentages = {label: round(100 * count / total_pixels, 2) for label, count in counts.items()}

    return percentages

def apply_damage_mask(frame, mask):
    colors = {
        0: (0, 255, 0),      # no damage → green
        1: (0, 255, 255),    # minor damage → yellow
        2: (0, 165, 255),    # major damage → orange
        3: (0, 0, 255)       # destroyed → red
    }

    mask_rgb = np.zeros_like(frame)
    for cls, color in colors.items():
        mask_rgb[mask == cls] = color

    blended = cv2.addWeighted(frame, 0.6, mask_rgb, 0.4, 0)
    return blended



def visualize_and_save(image, percentages, output_path, window_index=0):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Write percentages beside the image
    info = "\n".join([f"{label}: {pct}%" for label, pct in percentages.items()])
    plt.text(
        1.02, 0.5, info, transform=plt.gca().transAxes,
        fontsize=12, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1)
    )

    plt.tight_layout()
    plt.savefig(output_path)
    # Show the figure non-blocking
    mng = plt.get_current_fig_manager()
    try:
        # Offset window position (each new image slightly above previous)
        mng.window.wm_geometry(f"+100+{100 + window_index * 80}")
    except:
        pass  # Skip if backend doesn't support window repositioning
    plt.show(block=False)
    plt.pause(0.1)

CURRENT_ANALYSIS_FILE = os.path.join(os.path.dirname(__file__), "current_analysis.json")

def run_damage_analysis():
    if os.path.exists(CURRENT_ANALYSIS_FILE):
        os.remove(CURRENT_ANALYSIS_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = r"C:\Users\bolaky\PycharmProjects\Rebuild\Models\Dec21_11_50_seresnext50_unet_v2_512_fold3_fp16_crops.pth"
    model = load_model(checkpoint_path, device)
    #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #captured_folder = os.path.join("captured_images", timestamp)
    #analysis_folder = os.path.join("analysis_images", timestamp)
    #os.makedirs(analysis_folder, exist_ok=True)                    
    # Find the newest capture folder 
    base_captured_dir = r"C:\Users\bolaky\PycharmProjects\Rebuild\outputs\captured_images"
    if not os.path.exists(base_captured_dir) or not os.listdir(base_captured_dir):
        print(" No captured_images folders found. Please capture images first.")
        return 
    
    latest_folder = max(
        [os.path.join(base_captured_dir, d) for d in os.listdir(base_captured_dir)],
        key=os.path.getmtime
    )
    captured_folder = latest_folder

    #  Create a new analysis folder for this session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    analysis_folder = os.path.join("outputs/analysis_images", timestamp)
    os.makedirs(analysis_folder, exist_ok=True)

    processed = set()
    print(" Watching for new images...")
    window_index = 0

    while True:
        for filename in sorted_nicely(os.listdir(captured_folder)):
            if filename.endswith((".jpg", ".png")) and filename not in processed:
                image_path = os.path.join(captured_folder, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                print(f" Analyzing: {filename} ...")
                percentages = analyze_damage(model, image, device)
                #mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                #blended = apply_damage_mask(image, mask)
                
                print(f"DEBUG: Percentages before JSON write for {filename}: {percentages}")
                new_analysis_object = {
                    "filename": filename,
                    "percentages": percentages,
                }
                
                # 1. READ existing data (if any)
                current_data_list = []
                if os.path.exists(CURRENT_ANALYSIS_FILE) and os.path.getsize(CURRENT_ANALYSIS_FILE) > 0:
                    try:
                        with open(CURRENT_ANALYSIS_FILE, 'r') as f:
                            # Safely load the JSON. If it's empty or invalid, current_data_list remains empty.
                            current_data_list = json.load(f)
                            # Ensure the loaded data is a list, otherwise initialize an empty list
                            if not isinstance(current_data_list, list):
                                current_data_list = []
                    except json.JSONDecodeError:
                        # Handle case where file is corrupted or empty
                        current_data_list = []

                # 2. APPEND the new object to the list
                current_data_list.append(new_analysis_object)
                
                # 3. OVERWRITE the file with the complete, updated list
                with open(CURRENT_ANALYSIS_FILE, 'w') as f:
                    json.dump(current_data_list, f, indent=4) # Use 'w' to overwrite with the full list
                # Display and save
                output_path = os.path.join(analysis_folder, f"analysis_{filename}")
                visualize_and_save(image, percentages, output_path, window_index)

                window_index += 1
                processed.add(filename)

                # Wait 15 seconds before showing next one
                print(" Waiting 5 seconds before next analysis window...")
                time.sleep(5)

                processed.add(filename)

        time.sleep(2)  # Check every 2 seconds for new captures


if __name__ == "__main__":
    run_damage_analysis()



