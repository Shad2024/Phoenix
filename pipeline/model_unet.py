import torch
import torch.nn as nn

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