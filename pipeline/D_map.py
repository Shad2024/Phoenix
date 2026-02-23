import numpy as np
import torch
import matplotlib.pyplot as plt
from model_unet import UNet

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# Path to ONE training X file (.npz that contains "X" with shape H x W x 5)
TRAIN_X_FILE = r"C:\Users\bolaky\PycharmProjects\Rebuild\map_dataset\geoai_training_data_x\Luxembourg\X_Luxembourg_tile_0015.npz"

MODEL_PATH = "geoai_model.pth+"

# ---------------- LOAD TRAINING TILE ----------------
data = np.load(TRAIN_X_FILE)
X = data["X"]  # shape: (H, W, 5)

print("Loaded X shape:", X.shape)

# Convert to tensor exactly like training
x_tensor = torch.tensor(X, dtype=torch.float32)
x_tensor = x_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 5, H, W)
x_tensor = x_tensor.to(DEVICE)

print("Model input shape:", x_tensor.shape)

# ---------------- LOAD MODEL ----------------
model = UNet(in_ch=5, out_ch=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ---------------- INFERENCE ----------------
with torch.no_grad():
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)[0].cpu().numpy()

print("Prediction shape:", pred.shape)

# ---------------- DEBUG CLASS DISTRIBUTION ----------------
unique, counts = np.unique(pred, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# ---------------- VISUALIZE ----------------
plt.figure(figsize=(5, 5))
plt.imshow(pred, cmap="tab10")
plt.colorbar()
plt.title("Prediction on TRAINING tile")
plt.axis("off")
plt.show()
