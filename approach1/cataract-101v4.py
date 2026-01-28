import os
import random
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score


DATA_ROOT = "/Volumes/Extreme SSD/cataract-101"

CLIP_LEN = 16
FRAME_SUBSAMPLE = 15
BATCH_SIZE = 2
EPOCHS = 7
LR = 1e-4
VAL_RATIO = 0.2
RANDOM_SEED = 42
NUM_EVAL_CLIPS = 8
GRAD_CLIP_NORM = 5.0

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")


def build_meta(root: str) -> pd.DataFrame:
    csv_path = os.path.join(root, "videos.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"videos.csv not found at {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().replace(" ", "") for c in df.columns]

    expected_cols = {"VideoID", "Frames", "FPS", "Surgeon", "Experience"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Expected columns {expected_cols}, got {set(df.columns)}. "
            "Check the header of videos.csv."
        )

    df["Label"] = (df["Experience"].astype(int) - 1).clip(0, 1)
    return df


def stratified_split(meta_df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_ids = []
    val_ids = []

    for label in sorted(meta_df["Label"].unique()):
        vids = meta_df.loc[meta_df["Label"] == label, "VideoID"].values
        idxs = np.arange(len(vids))
        rng.shuffle(idxs)
        cut = int(len(vids) * (1.0 - val_ratio))
        train_ids.extend(vids[idxs[:cut]].tolist())
        val_ids.extend(vids[idxs[cut:]].tolist())

    return train_ids, val_ids


def compute_class_weights(meta_df: pd.DataFrame) -> torch.Tensor:
    label_counts = meta_df["Label"].value_counts().to_dict()
    num_classes = len(label_counts)
    total = len(meta_df)

    weights = []
    for c in range(num_classes):
        n_c = label_counts.get(c, 1)
        w_c = total / (num_classes * n_c)
        weights.append(w_c)

    return torch.tensor(weights, dtype=torch.float32)


def compute_max_start(num_frames: int, clip_len: int, frame_subsample: int) -> int:
    return max(0, num_frames - frame_subsample * (clip_len - 1) - 1)


def make_deterministic_eval_starts(
    val_df: pd.DataFrame,
    num_eval_clips: int,
    clip_len: int,
    frame_subsample: int,
    seed: int,
) -> Dict[int, List[int]]:
    starts = {}
    for _, row in val_df.iterrows():
        vid = int(row["VideoID"])
        num_frames = int(row["Frames"])
        max_start = compute_max_start(num_frames, clip_len, frame_subsample)

        if max_start == 0:
            starts[vid] = [0] * num_eval_clips
            continue

        local_rng = np.random.default_rng(seed + vid * 10007)
        cand = local_rng.integers(low=0, high=max_start + 1, size=num_eval_clips).tolist()
        starts[vid] = cand
    return starts


class CataractSkillDataset(Dataset):
    def __init__(
        self,
        root: str,
        video_ids,
        meta_df: pd.DataFrame,
        clip_len: int = 16,
        frame_subsample: int = 15,
        transform=None,
    ):
        self.root = root
        self.videos_dir = os.path.join(root, "videos")
        self.clip_len = clip_len
        self.frame_subsample = frame_subsample
        self.transform = transform

        self.df = meta_df[meta_df["VideoID"].isin(video_ids)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No videos found for given VideoIDs subset.")

    def __len__(self):
        return len(self.df)

    def _indices_from_start(self, num_frames: int, start: int) -> List[int]:
        idxs = [start + i * self.frame_subsample for i in range(self.clip_len)]
        idxs = [min(i, num_frames - 1) for i in idxs]
        return idxs

    def _random_start(self, num_frames: int) -> int:
        max_start = compute_max_start(num_frames, self.clip_len, self.frame_subsample)
        return random.randint(0, max_start) if max_start > 0 else 0

    def _read_frames(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                    continue
                cap.release()
                raise IOError(f"Failed to read frame {idx} from {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            frames.append(frame)

        cap.release()
        return torch.stack(frames, dim=0)

    def get_clip_by_start(self, video_id: int, num_frames: int, start: int) -> torch.Tensor:
        video_filename = f"case_{video_id}.mp4"
        video_path = os.path.join(self.videos_dir, video_filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        frame_indices = self._indices_from_start(num_frames, start)
        return self._read_frames(video_path, frame_indices)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_id = int(row["VideoID"])
        num_frames = int(row["Frames"])
        label = int(row["Label"])

        start = self._random_start(num_frames)
        clip = self.get_clip_by_start(video_id, num_frames, start)

        return clip, torch.tensor(label, dtype=torch.long)


class EffNetLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        self.feature_extractor = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),
        )

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        return self.classifier(last)


def unfreeze_backbone(model: EffNetLSTM):
    for p in model.feature_extractor.parameters():
        p.requires_grad = True


def collate_fn(batch):
    clips, labels = zip(*batch)
    return torch.stack(clips, dim=0), torch.stack(labels, dim=0)


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (clips, labels) in enumerate(loader):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i % 10 == 0:
            print(f"[Epoch {epoch+1} | Step {i}] loss = {loss.item():.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_video_level_auc_acc(model, val_ds: CataractSkillDataset, starts_map: Dict[int, List[int]]):
    model.eval()
    y_true = []
    y_score = []

    for i in range(len(val_ds.df)):
        row = val_ds.df.iloc[i]
        vid = int(row["VideoID"])
        num_frames = int(row["Frames"])
        label = int(row["Label"])

        clip_probs = []
        for start in starts_map[vid]:
            clip = val_ds.get_clip_by_start(vid, num_frames, int(start)).unsqueeze(0).to(device)
            logits = model(clip)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            clip_probs.append(prob)

        y_true.append(label)
        y_score.append(float(np.mean(clip_probs)))

    y_true = np.array(y_true, dtype=np.int64)
    y_score = np.array(y_score, dtype=np.float32)
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    return auc, acc


def make_optimizer_and_scheduler(model: nn.Module, lr: float):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=1e-3,
        verbose=True,
        min_lr=1e-6,
    )
    return optimizer, scheduler


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    meta = build_meta(DATA_ROOT)
    print(meta.head())

    train_ids, val_ids = stratified_split(meta, val_ratio=VAL_RATIO, seed=RANDOM_SEED)
    print(f"Train videos: {len(train_ids)}, Val videos: {len(val_ids)}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = CataractSkillDataset(
        DATA_ROOT, train_ids, meta,
        clip_len=CLIP_LEN,
        frame_subsample=FRAME_SUBSAMPLE,
        transform=transform,
    )
    val_ds = CataractSkillDataset(
        DATA_ROOT, val_ids, meta,
        clip_len=CLIP_LEN,
        frame_subsample=FRAME_SUBSAMPLE,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = EffNetLSTM(num_layers=2).to(device)

    class_weights = compute_class_weights(train_ds.df).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer, scheduler = make_optimizer_and_scheduler(model, LR)

    starts_map = make_deterministic_eval_starts(
        val_ds.df, NUM_EVAL_CLIPS, CLIP_LEN, FRAME_SUBSAMPLE, RANDOM_SEED
    )

    best_val_auc = -1.0

    for epoch in range(EPOCHS):
        if epoch == 1:
            print("🔓 Unfreezing EfficientNet backbone for fine-tuning")
            unfreeze_backbone(model)
            optimizer, scheduler = make_optimizer_and_scheduler(model, LR * 0.5)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

        val_auc, val_acc = eval_video_level_auc_acc(model, val_ds, starts_map)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val AUC: {val_auc:.3f}, Val acc: {val_acc:.3f}"
        )

        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "effnet_lstm_cataract101_best.pt")
            print(f"  🔹 New best model saved (val AUC = {val_auc:.3f})")

    print("Training complete.")


@torch.no_grad()
def evaluate_auc_acc():
    meta = build_meta(DATA_ROOT)
    train_ids, val_ids = stratified_split(meta, val_ratio=VAL_RATIO, seed=RANDOM_SEED)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_ds = CataractSkillDataset(
        DATA_ROOT, val_ids, meta,
        clip_len=CLIP_LEN,
        frame_subsample=FRAME_SUBSAMPLE,
        transform=transform,
    )

    starts_map = make_deterministic_eval_starts(
        val_ds.df, NUM_EVAL_CLIPS, CLIP_LEN, FRAME_SUBSAMPLE, RANDOM_SEED
    )

    model = EffNetLSTM(num_layers=2).to(device)
    model.load_state_dict(torch.load("effnet_lstm_cataract101_best.pt", map_location=device))
    model.eval()

    auc, acc = eval_video_level_auc_acc(model, val_ds, starts_map)
    print(f"Deterministic Video-level Validation AUC: {auc:.3f}")
    print(f"Deterministic Video-level Validation Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
    evaluate_auc_acc()
