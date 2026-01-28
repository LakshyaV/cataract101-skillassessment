import os
import re
import csv
import glob
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os
import re
import csv
import glob
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -------------------------
# Repro / device
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Losses
# -------------------------
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(0, 2, 3))
    den = (probs + targets).sum(dim=(0, 2, 3)) + eps
    dice = 1 - (num / den)
    return dice.mean()


def seg_loss(logits, targets, dice_w=1.0, bce_w=1.0):
    return dice_w * dice_loss_with_logits(logits, targets) + bce_w * F.binary_cross_entropy_with_logits(logits, targets)


# -------------------------
# Model components
# -------------------------
class TemporalAttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor):
        w = self.score(h).squeeze(-1)
        w = torch.softmax(w, dim=1)
        z = (h * w.unsqueeze(-1)).sum(dim=1)
        return z, w


class EffB0FeatureMaps(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=weights)
        self.features = m.features

    def forward(self, x):
        return self.features(x)


class WetCatPerception(nn.Module):
    def __init__(self, num_phases: int, pretrained_backbone=True):
        super().__init__()
        self.backbone = EffB0FeatureMaps(pretrained=pretrained_backbone)

        self.seg_head = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1),
        )

        self.phase_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_phases),
        )

        self.attn_head = nn.Sequential(
            nn.Conv2d(1280, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x2 = x.reshape(B * T, C, H, W)
            Fm = self.backbone(x2)
            seg = self.seg_head(Fm)
            att = self.attn_head(Fm)

            ph = self.phase_head(Fm)
            ph = ph.reshape(B, T, -1).mean(dim=1)
            return Fm, seg, att, ph

        Fm = self.backbone(x)
        seg = self.seg_head(Fm)
        att = self.attn_head(Fm)
        ph = self.phase_head(Fm)
        return Fm, seg, att, ph


class SkillModel(nn.Module):
    def __init__(self, perception: WetCatPerception, temporal: str = "lstm", hidden: int = 512, num_layers: int = 1):
        super().__init__()
        self.perception = perception
        self.temporal = temporal

        if temporal == "lstm":
            self.rnn = nn.LSTM(
                input_size=1280,
                hidden_size=hidden,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
            )
            out_dim = hidden * 2
        elif temporal == "transformer":
            enc_layer = nn.TransformerEncoderLayer(d_model=1280, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            out_dim = 1280
        else:
            raise ValueError("temporal must be 'lstm' or 'transformer'")

        self.pool = TemporalAttnPool(out_dim)
        self.skill_head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    @torch.no_grad()
    def _infer_attention_from_seg(self, seg_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(seg_logits)
        union = torch.clamp(probs.sum(dim=1, keepdim=True), 0, 1)
        return union

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape

        Fm, seg, _att_head, _ph = self.perception(x)
        att = self._infer_attention_from_seg(seg)
        att = torch.sigmoid(att)

        Fm_att = Fm * att
        emb = F.adaptive_avg_pool2d(Fm_att, 1).squeeze(-1).squeeze(-1)
        emb = emb.reshape(B, T, 1280)

        if self.temporal == "lstm":
            h, _ = self.rnn(emb)
        else:
            h = self.transformer(emb)

        z, w = self.pool(h)
        logit = self.skill_head(z).squeeze(-1)
        return logit, w


# -------------------------
# WetCat utilities (phases + masks)
# -------------------------
def read_phase_csv(csv_path: str) -> List[Tuple[float, float, str]]:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((float(r["start"]), float(r["end"]), str(r["phase"])))
    return rows


def phase_at_time(phase_rows: List[Tuple[float, float, str]], t_sec: float) -> str:
    for s, e, p in phase_rows:
        if s <= t_sec < e:
            return p
    return phase_rows[-1][2] if phase_rows else "unknown"


def load_mask_png_binary(path: str, out_h: int, out_w: int) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        arr = (arr > 128).astype(np.uint8)
        img2 = Image.fromarray(arr * 255).resize((out_w, out_h), resample=Image.NEAREST)
        arr2 = (np.array(img2) > 128).astype(np.float32)
        return torch.from_numpy(arr2)


def infer_wetcat_ids(phase_dir: str) -> List[str]:
    ids = []
    for fn in os.listdir(phase_dir):
        m = re.match(r"wetlab_cataract_(\d+)_phases\.csv", fn)
        if m:
            ids.append(m.group(1))
    return sorted(ids)


def build_phase_vocab(phase_dir: str, ids: List[str]) -> List[str]:
    vocab = set()
    for vid in ids:
        p = os.path.join(phase_dir, f"wetlab_cataract_{vid}_phases.csv")
        if not os.path.exists(p):
            continue
        rows = read_phase_csv(p)
        for _, _, name in rows:
            vocab.add(name)
    vocab = sorted(list(vocab))
    if not vocab:
        vocab = ["unknown"]
    return vocab


def find_video_file(videos_dir: str, vid: str) -> str:
    patterns = [
        os.path.join(videos_dir, f"wetlab_cataract_{vid}.*"),
        os.path.join(videos_dir, f"wetlab_cataract_{vid.zfill(3)}.*"),
        os.path.join(videos_dir, f"*{vid}*.mp4"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"No video found for vid={vid} in {videos_dir}")


def available_mask_indices(mask_folder: str, vid: str, cls_name: str) -> List[int]:
    pat = os.path.join(mask_folder, f"wetlab_cataract_{vid}_*_{cls_name}_mask.png")
    hits = glob.glob(pat)
    idxs = []
    for h in hits:
        base = os.path.basename(h)
        m = re.match(rf"wetlab_cataract_{re.escape(vid)}_(\d+)_"+re.escape(cls_name)+r"_mask\.png", base)
        if m:
            idxs.append(int(m.group(1)))
        else:
            mm = re.search(r"_(\d{4})_", base)
            if mm:
                idxs.append(int(mm.group(1)))
    return sorted(list(set(idxs)))


def nearest_index(sorted_indices: List[int], target: int) -> int:
    if not sorted_indices:
        return target
    import bisect
    pos = bisect.bisect_left(sorted_indices, target)
    if pos == 0:
        return sorted_indices[0]
    if pos == len(sorted_indices):
        return sorted_indices[-1]
    before = sorted_indices[pos - 1]
    after = sorted_indices[pos]
    return before if abs(before - target) <= abs(after - target) else after


# -------------------------
# Datasets
# -------------------------
class WetCatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str,
        clip_len: int = 16,
        stride: int = 4,
        resize: Tuple[int, int] = (224, 224),
        ids: Optional[List[str]] = None,
        phase_vocab: Optional[List[str]] = None,
    ):
        self.base_dir = base_dir
        self.videos_dir = os.path.join(base_dir, "videos")
        self.phase_dir = os.path.join(base_dir, "Phase_Annotations")
        self.masks_dir = os.path.join(base_dir, "Segmentation_Masks")

        self.clip_len = clip_len
        self.stride = stride
        self.H, self.W = resize

        if ids is None:
            ids = infer_wetcat_ids(self.phase_dir)
        self.ids = ids

        if phase_vocab is None:
            phase_vocab = build_phase_vocab(self.phase_dir, self.ids)
        self.phase_vocab = phase_vocab
        self.phase_to_idx = {p: i for i, p in enumerate(self.phase_vocab)}

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.H, self.W)),
            torchvision.transforms.ToTensor(),
        ])

        self.mask_index_cache: Dict[Tuple[str, str], List[int]] = {}
        for vid in self.ids:
            for cls_folder, cls_name in [
                ("Instruments", "instruments"),
                ("Pupil", "pupil"),
                ("Iris", "iris"),
                ("Rhexis", "rhexis"),
            ]:
                folder = os.path.join(self.masks_dir, cls_folder)
                self.mask_index_cache[(vid, cls_name)] = available_mask_indices(folder, vid, cls_name)

    def __len__(self):
        return len(self.ids)

    def _mask_path(self, cls_folder: str, vid: str, frame_idx_1based: int, cls_name: str) -> str:
        return os.path.join(
            self.masks_dir, cls_folder,
            f"wetlab_cataract_{vid}_{frame_idx_1based:04d}_{cls_name}_mask.png"
        )

    def __getitem__(self, i: int):
        vid = self.ids[i]
        vpath = find_video_file(self.videos_dir, vid)

        phase_path = os.path.join(self.phase_dir, f"wetlab_cataract_{vid}_phases.csv")
        phase_rows = read_phase_csv(phase_path)

        frames, _, info = torchvision.io.read_video(vpath, pts_unit="sec")
        fps = float(info.get("video_fps", 30.0))
        N = frames.shape[0]

        max_start = N - (self.clip_len * self.stride)
        start = 0 if max_start <= 0 else random.randint(0, max_start)
        idxs = [start + k * self.stride for k in range(self.clip_len)]

        clip_frames = []
        clip_masks = []
        clip_phase_idxs = []

        for fi in idxs:
            img = Image.fromarray(frames[fi].numpy())
            x = self.transform(img)
            clip_frames.append(x)

            target_mask_idx = fi + 1

            masks_4 = []
            for cls_folder, cls_name in [
                ("Instruments", "instruments"),
                ("Pupil", "pupil"),
                ("Iris", "iris"),
                ("Rhexis", "rhexis"),
            ]:
                avail = self.mask_index_cache.get((vid, cls_name), [])
                use_idx = nearest_index(avail, target_mask_idx)
                mp = self._mask_path(cls_folder, vid, use_idx, cls_name)
                if not os.path.exists(mp):
                    gp = glob.glob(os.path.join(self.masks_dir, cls_folder, f"wetlab_cataract_{vid}_*_{cls_name}_mask.png"))
                    if not gp:
                        raise FileNotFoundError(f"Mask not found for vid={vid} cls={cls_name} in {self.masks_dir}/{cls_folder}")
                    mp = sorted(gp)[0]
                m = load_mask_png_binary(mp, self.H, self.W)
                masks_4.append(m)
            m4 = torch.stack(masks_4, dim=0)
            clip_masks.append(m4)

            t_sec = fi / fps
            pname = phase_at_time(phase_rows, t_sec)
            clip_phase_idxs.append(self.phase_to_idx.get(pname, 0))

        clip_frames = torch.stack(clip_frames, dim=0)
        clip_masks = torch.stack(clip_masks, dim=0)

        phase = int(np.bincount(np.array(clip_phase_idxs)).argmax()) if clip_phase_idxs else 0
        return clip_frames, clip_masks, torch.tensor(phase, dtype=torch.long)


class Cataract101Dataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Tuple[str, float]], clip_len=16, stride=4, resize=(224, 224)):
        self.items = items
        self.clip_len = clip_len
        self.stride = stride
        self.H, self.W = resize
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.H, self.W)),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vpath, y = self.items[i]
        frames, _, info = torchvision.io.read_video(vpath, pts_unit="sec")
        N = frames.shape[0]

        max_start = N - (self.clip_len * self.stride)
        start = 0 if max_start <= 0 else random.randint(0, max_start)
        idxs = [start + k * self.stride for k in range(self.clip_len)]

        clip = []
        for fi in idxs:
            img = Image.fromarray(frames[fi].numpy())
            clip.append(self.transform(img))
        clip = torch.stack(clip, dim=0)

        return clip, torch.tensor(y, dtype=torch.float32)


def load_c101_items(base_dir: str) -> List[Tuple[str, float]]:
    csv_path = os.path.join(base_dir, "cataract101_labels.csv")
    if os.path.exists(csv_path):
        items = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                vp = r["video_path"]
                lab = float(r["label"])
                if lab in (1.0, 2.0):
                    lab = 0.0 if lab == 1.0 else 1.0
                items.append((vp, lab))
        if items:
            return items

    root = os.path.join(base_dir, "cataract101", "videos")
    exp1 = glob.glob(os.path.join(root, "Experience_1", "*.*"))
    exp2 = glob.glob(os.path.join(root, "Experience_2", "*.*"))
    items = []
    for vp in exp1:
        items.append((vp, 0.0))
    for vp in exp2:
        items.append((vp, 1.0))
    return items


# -------------------------
# Training helpers
# -------------------------
def union_mask_from_targets(masks_bt: torch.Tensor) -> torch.Tensor:
    return torch.clamp(masks_bt.sum(dim=1, keepdim=True), 0, 1)


@dataclass
class TrainConfig:
    base_dir: str
    clip_len: int = 16
    stride: int = 4
    resize: Tuple[int, int] = (224, 224)
    batch_size: int = 2
    num_workers: int = 2
    seed: int = 42


def train_wetcat(
    cfg: TrainConfig,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    phase_loss_w: float = 0.3,
    attn_loss_w: float = 0.2,
    dice_w: float = 1.0,
    bce_w: float = 1.0,
    save_path: str = "wetcat_pretrained_backbone.pt",
):
    device = get_device()
    set_seed(cfg.seed)

    ids = infer_wetcat_ids(os.path.join(cfg.base_dir, "Phase_Annotations"))
    phase_vocab = build_phase_vocab(os.path.join(cfg.base_dir, "Phase_Annotations"), ids)

    ds = WetCatDataset(
        base_dir=cfg.base_dir,
        clip_len=cfg.clip_len,
        stride=cfg.stride,
        resize=cfg.resize,
        ids=ids,
        phase_vocab=phase_vocab,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False
    )

    model = WetCatPerception(num_phases=len(phase_vocab), pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[WetCat] device={device} | num_ids={len(ds)} | phases={phase_vocab}")

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for frames, masks, phase in dl:
            frames = frames.to(device)
            masks = masks.to(device)
            phase = phase.to(device)

            B, T, _, H, W = frames.shape
            Fm, seg, attn, ph = model(frames)

            h, w = seg.shape[-2:]
            masks_bt = masks.view(B * T, 4, H, W)
            masks_ds = F.interpolate(masks_bt, size=(h, w), mode="nearest")

            Lseg = dice_w * dice_loss_with_logits(seg, masks_ds) + bce_w * F.binary_cross_entropy_with_logits(seg, masks_ds)
            Lph = F.cross_entropy(ph, phase)

            attn_tgt = union_mask_from_targets(masks_ds)
            Lattn = F.binary_cross_entropy_with_logits(attn, attn_tgt)

            loss = Lseg + phase_loss_w * Lph + attn_loss_w * Lattn

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())

        avg = running / max(len(dl), 1)
        print(f"[WetCat] epoch {epoch+1}/{epochs} loss={avg:.4f}")

        torch.save(
            {
                "state_dict": model.state_dict(),
                "phase_vocab": phase_vocab,
                "cfg": {
                    "clip_len": cfg.clip_len,
                    "stride": cfg.stride,
                    "resize": cfg.resize,
                },
            },
            save_path,
        )

    print(f"[WetCat] saved -> {save_path}")


def train_c101(
    cfg: TrainConfig,
    wetcat_ckpt: str = "wetcat_pretrained_backbone.pt",
    temporal: str = "lstm",
    hidden: int = 512,
    epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    freeze_perception: bool = True,
    save_path: str = "c101_skill_best.pt",
    train_split: float = 0.8,
):
    device = get_device()
    set_seed(cfg.seed)

    items = load_c101_items(cfg.base_dir)
    items = [(vp, y) for vp, y in items if os.path.exists(vp)]
    if len(items) < 2:
        raise RuntimeError(
            "Cataract-101 items not found.\n"
            "Provide cataract101_labels.csv in cataract_dataset OR create folder structure:\n"
            "  cataract_dataset/cataract101/videos/Experience_1/*.mp4\n"
            "  cataract_dataset/cataract101/videos/Experience_2/*.mp4\n"
        )

    random.shuffle(items)
    n_train = int(len(items) * train_split)
    train_items = items[:n_train]
    val_items = items[n_train:] if n_train < len(items) else items[: max(1, len(items)//5)]

    train_ds = Cataract101Dataset(train_items, clip_len=cfg.clip_len, stride=cfg.stride, resize=cfg.resize)
    val_ds = Cataract101Dataset(val_items, clip_len=cfg.clip_len, stride=cfg.stride, resize=cfg.resize)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    ckpt = torch.load(wetcat_ckpt, map_location="cpu")
    phase_vocab = ckpt.get("phase_vocab", ["unknown"])

    perception = WetCatPerception(num_phases=len(phase_vocab), pretrained_backbone=False)
    perception.load_state_dict(ckpt["state_dict"], strict=True)
    perception.to(device)

    if freeze_perception:
        for p in perception.parameters():
            p.requires_grad = False
        perception.eval()

    model = SkillModel(perception=perception, temporal=temporal, hidden=hidden).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    print(f"[C101] device={device} | items={len(items)} | train={len(train_items)} val={len(val_items)} | temporal={temporal}")

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0

        for clip, y in train_dl:
            clip = clip.to(device)
            y = y.to(device)

            logit, _w = model(clip)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += float(loss.item())

        tr_loss /= max(len(train_dl), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for clip, y in val_dl:
                clip = clip.to(device)
                y = y.to(device)
                logit, _ = model(clip)
                va_loss += float(F.binary_cross_entropy_with_logits(logit, y).item())
        va_loss /= max(len(val_dl), 1)

        print(f"[C101] epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "temporal": temporal,
                    "hidden": hidden,
                    "cfg": {
                        "clip_len": cfg.clip_len,
                        "stride": cfg.stride,
                        "resize": cfg.resize,
                    },
                },
                save_path,
            )

    print(f"[C101] saved best -> {save_path}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["wetcat", "c101"], required=True)

    ap.add_argument("--base_dir", type=str, default="/Volumes/Extreme SSD/cataract_dataset")

    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--resize", type=int, nargs=2, default=[224, 224])

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # WetCat args
    ap.add_argument("--wetcat_epochs", type=int, default=10)
    ap.add_argument("--wetcat_lr", type=float, default=1e-4)
    ap.add_argument("--wetcat_wd", type=float, default=1e-2)
    ap.add_argument("--phase_loss_w", type=float, default=0.3)
    ap.add_argument("--attn_loss_w", type=float, default=0.2)
    ap.add_argument("--wetcat_save", type=str, default="wetcat_pretrained_backbone.pt")

    # C101 args
    ap.add_argument("--c101_epochs", type=int, default=10)
    ap.add_argument("--c101_lr", type=float, default=2e-4)
    ap.add_argument("--c101_wd", type=float, default=1e-2)
    ap.add_argument("--wetcat_ckpt", type=str, default="wetcat_pretrained_backbone.pt")
    ap.add_argument("--temporal", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--freeze_perception", action="store_true")
    ap.add_argument("--c101_save", type=str, default="c101_skill_best.pt")
    ap.add_argument("--train_split", type=float, default=0.8)

    args = ap.parse_args()

    cfg = TrainConfig(
        base_dir=args.base_dir,
        clip_len=args.clip_len,
        stride=args.stride,
        resize=(args.resize[0], args.resize[1]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.stage == "wetcat":
        train_wetcat(
            cfg=cfg,
            epochs=args.wetcat_epochs,
            lr=args.wetcat_lr,
            weight_decay=args.wetcat_wd,
            phase_loss_w=args.phase_loss_w,
            attn_loss_w=args.attn_loss_w,
            save_path=args.wetcat_save,
        )

    elif args.stage == "c101":
        train_c101(
            cfg=cfg,
            wetcat_ckpt=args.wetcat_ckpt,
            temporal=args.temporal,
            hidden=args.hidden,
            epochs=args.c101_epochs,
            lr=args.c101_lr,
            weight_decay=args.c101_wd,
            freeze_perception=args.freeze_perception,
            save_path=args.c101_save,
            train_split=args.train_split,
        )


if __name__ == "__main__":
    main()