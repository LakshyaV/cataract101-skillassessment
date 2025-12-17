import os
import glob
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PIL import Image
from natsort import natsorted
import pandas as pd
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "/Volumes/Extreme SSD/cataract_dataset"
FRAMES_DIR = os.path.join(DATA_ROOT, "frames")
LABELS_CSV = os.path.join(DATA_ROOT, "skill_labels.csv")  # <-- make this file
CLIP_LEN = 32
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# -------------------------
# UTIL: DEVICE
# -------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# -------------------------
# LABEL LOADING
# -------------------------
def load_labels(csv_path):
    """
    CSV with columns: video_id,label
    label ∈ {0,1}
    """
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row["video_id"])] = int(row["label"])
    return labels

# -------------------------
# DATASET
# -------------------------
class CataractClipDataset(Dataset):
    def __init__(self, frames_dir, labels_dict, clip_len=32, transform=None):
        super().__init__()
        self.frames_dir = frames_dir
        self.labels_dict = labels_dict
        self.clip_len = clip_len
        self.transform = transform

        # Get all frame files
        all_frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
        all_frames += glob.glob(os.path.join(frames_dir, "*.png"))

        if len(all_frames) == 0:
            raise RuntimeError(f"No frame images found in {frames_dir}")

        # Group frames by video id (prefix like wetlab_cataract_001)
        self.video_to_frames = {}
        for f in all_frames:
            fname = os.path.basename(f)
            parts = fname.split("_")
            # e.g. wetlab_cataract_001_0001.jpg -> wetlab_cataract_001
            video_id = "_".join(parts[:3])
            # only keep videos that have labels
            if video_id not in labels_dict:
                continue
            self.video_to_frames.setdefault(video_id, []).append(f)

        # Sort frames within each video
        for vid in self.video_to_frames:
            self.video_to_frames[vid] = natsorted(self.video_to_frames[vid])

        self.video_ids = sorted(self.video_to_frames.keys())

        if len(self.video_ids) == 0:
            raise RuntimeError(
                "No videos found that match your labels. "
                "Check that video_id in the CSV matches the filename prefix."
            )

        print(f"Found {len(self.video_ids)} labeled videos.")

    def __len__(self):
        return len(self.video_ids)

    def _sample_clip_paths(self, frames):
        """
        Randomly sample a contiguous clip of length clip_len.
        Loop-pad if too short.
        """
        if len(frames) < self.clip_len:
            reps = (self.clip_len + len(frames) - 1) // len(frames)
            frames = (frames * reps)[: self.clip_len]

        if len(frames) == self.clip_len:
            return frames

        start = random.randint(0, len(frames) - self.clip_len)
        return frames[start : start + self.clip_len]

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = self.video_to_frames[video_id]
        clip_paths = self._sample_clip_paths(frames)

        imgs = []
        for fp in clip_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        clip_tensor = torch.stack(imgs, dim=0)  # (T, C, H, W)
        label = torch.tensor(self.labels_dict[video_id], dtype=torch.float32)
        return clip_tensor, label

# -------------------------
# MODEL: EfficientNet-B0 + LSTM
# -------------------------
class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280
        # store normalization transform for dataloader
        self.preprocess = weights.transforms()

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.encoder = FrameEncoder()
        feature_dim = self.encoder.out_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)          # (B*T, F)
        feats = feats.view(B, T, -1)     # (B, T, F)
        lstm_out, _ = self.lstm(feats)   # (B, T, H)
        last = lstm_out[:, -1, :]        # (B, H)
        logits = self.classifier(last)   # (B, 1)
        return logits.squeeze(1)

# -------------------------
# TRAIN / EVAL
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Train", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * clips.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Val", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips)
        loss = criterion(logits, labels)

        running_loss += loss.item() * clips.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# -------------------------
# MAIN
# -------------------------
def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    labels_dict = load_labels(LABELS_CSV)

    # temporary encoder instance just to get the transform
    temp_encoder = FrameEncoder()
    preprocess = temp_encoder.preprocess  # includes resize + normalize

    dataset = CataractClipDataset(
        frames_dir=FRAMES_DIR,
        labels_dict=labels_dict,
        clip_len=CLIP_LEN,
        transform=preprocess,
    )

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNNLSTM(hidden_dim=512).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_lstm_baseline.pt")
            print(f"✅ Saved new best model (val_acc={val_acc:.4f})")

if __name__ == "__main__":
    main()