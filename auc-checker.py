from sklearn.metrics import roc_auc_score, accuracy_score
import os
import random
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


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
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = EffNetLSTM().to(device)
    model.load_state_dict(torch.load("effnet_lstm_cataract101_best.pt", map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    for clips, labels in val_loader:
        clips = clips.to(device)
        labels = labels.to(device)
        logits = model(clips)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)

    print(f"Validation AUC: {auc:.3f}")
    print(f"Validation Accuracy: {acc:.3f}")

if __name__ == "__main__":
    evaluate_auc_acc()