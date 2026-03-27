import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# DATASET
# =========================
class MRIDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        # Expected format:
        # kspace: (H, W, 2)  -> real + imag
        # target: (H, W)
        kspace = data["kspace"]
        target = data["target"]

        kspace = torch.tensor(kspace).permute(2, 0, 1).float()
        target = torch.tensor(target).unsqueeze(0).float()

        # Normalize
        kspace = kspace / (kspace.abs().max() + 1e-8)
        target = target / (target.max() + 1e-8)

        return kspace, target


# =========================
# DATA CONSISTENCY
# =========================
def data_consistency(image, kspace, mask=None):
    image_k = torch.fft.fft2(image)

    if mask is None:
        return image

    image_k = mask * kspace + (1 - mask) * image_k
    image = torch.fft.ifft2(image_k).real
    return image


# =========================
# TRM BLOCK (Recursive Unit)
# =========================
class TRMBlock(nn.Module):
    def __init__(self, channels=2, hidden=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, image, kspace):
        x = torch.cat([image, kspace[:, :1]], dim=1)  # use real part
        return self.net(x)


# =========================
# TRM MODEL (Recursive)
# =========================
class TRM_MRI(nn.Module):
    def __init__(self, steps=6):
        super().__init__()
        self.steps = steps
        self.block = TRMBlock()

    def forward(self, kspace):
        # initial reconstruction (zero-filled)
        image = torch.fft.ifft2(
            torch.complex(kspace[:, 0], kspace[:, 1])
        ).real.unsqueeze(1)

        for _ in range(self.steps):
            update = self.block(image, kspace)
            image = image + update
            image = data_consistency(image, kspace)

        return image


# =========================
# METRICS
# =========================
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


# =========================
# TRAINING LOOP
# =========================
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for kspace, target in dataloader:
        kspace = kspace.to(device)
        target = target.to(device)

        pred = model(kspace)

        loss = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = 0

    with torch.no_grad():
        for kspace, target in dataloader:
            kspace = kspace.to(device)
            target = target.to(device)

            pred = model(kspace)
            total_psnr += psnr(pred, target).item()

    return total_psnr / len(dataloader)


# =========================
# MAIN
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: replace with your dataset paths
    train_files = ["data/sample1.npy", "data/sample2.npy"]
    val_files = ["data/sample1.npy"]

    train_ds = MRIDataset(train_files)
    val_ds = MRIDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = TRM_MRI(steps=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        loss = train(model, train_loader, optimizer, device)
        val_psnr = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Loss={loss:.4f}, PSNR={val_psnr:.2f}")


if __name__ == "__main__":
    main()
