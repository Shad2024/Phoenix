import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import glob
import random
import sys
import os

dataset_path = r"C:\Users\bolaky\PycharmProjects\Rebuild\map_dataset"
sys.path.append(dataset_path)


class GeoAIDataset(Dataset):
    def __init__(self, x_folder, y_folder):
        self.x_files = sorted(glob.glob(x_folder + "/*.npz"))
        self.y_files = sorted(glob.glob(y_folder + "/*.npz"))

        # Load ALL Y tensors into memory (small)
        self.y_tensors = []
        for f in self.y_files:
            data = np.load(f)["label"]
            self.y_tensors.append(torch.tensor(data, dtype=torch.float32))

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
    # Load X = 5-channel constraints
       x_data = np.load(self.x_files[idx])["X"]
       x_tensor = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)

    # Randomly pick one Y style (unpaired)
       y_tensor = random.choice(self.y_tensors)
    
    # If Y is 2D, add channel dimension
       if y_tensor.dim() == 2:
           y_tensor = y_tensor.unsqueeze(-1)  # H x W -> H x W x 1

       y_tensor = y_tensor.permute(2, 0, 1)  # Now safe: 1 x H x W

       y_tensor = F.interpolate(
       y_tensor.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False
         ).squeeze(0)

       return x_tensor, y_tensor


# -----------------------
# 2. Model (Simple U-Net)
# -----------------------


class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=4):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = C(5, 32)
        self.enc2 = C(32, 64)
        self.enc3 = C(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = C(128, 64)
        self.dec2 = C(64, 32)
        self.final = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        print("Input:", x.shape)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = self.dec3(torch.nn.functional.interpolate(
            e3, size=e2.shape[2:], mode='bilinear', align_corners=False))
        d2 = self.dec2(torch.nn.functional.interpolate(
            d3, size=e1.shape[2:], mode='bilinear', align_corners=False))

        out = self.final(d2)
        print("Output:", out.shape)

        return out


# -----------------------
# 3. Training Loop
# -----------------------
def train():

    x_folder = r"C:\Users\bolaky\PycharmProjects\Rebuild\map_dataset\geoai_training_data_x\Luxembourg"
    y_folder = r"C:\Users\bolaky\PycharmProjects\Rebuild\map_dataset\geoai_training_data_y+"

    import glob
    x_files = glob.glob(x_folder + "/*.npz")
    y_files = glob.glob(y_folder + "/*.npz")
    print(f"Found {len(x_files)} X files and {len(y_files)} Y files")
    if len(x_files) == 0 or len(y_files) == 0:
        raise ValueError("Check folder paths or file names!")

    dataset = GeoAIDataset(x_folder, y_folder)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().cuda()
    criterion = nn.L1Loss()   # MAE works best for images
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(40):
        total_loss = 0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "geoai_model.pth+")
    print("\nModel saved as geoai_model.pth")


if __name__ == "__main__":
    train()
