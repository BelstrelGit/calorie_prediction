
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from scripts.dataset import get_dataloaders
from scripts.model import CalorieModel



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(cfg):
    set_seed(cfg.SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    train_data_loader, val_data_loader, _ = get_dataloaders(cfg)
    print("Train samples:", len(train_data_loader.dataset))
    print("Validation samples:", len(val_data_loader.dataset))
    batch = next(iter(train_data_loader))

    print("Images:", batch["image"].shape)
    print("Mass:", batch["mass"].shape)
    print("Targets:", batch["target"].shape)
    print("Texts:", len(batch["text"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

    model = CalorieModel(cfg)
    model = model.to(device)

    criterion = nn.L1Loss() #MAE |prediction - target|

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.HEAD_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.HEAD_LR},
        {"params": model.regression_head.parameters(), "lr": cfg.HEAD_LR}]
    )

    best_val_mae = float("inf")
    history = {
        "train_mae": [],
        "val_mae": []
    }
    os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_data_loader:
            texts = batch["text"]

            encoded_text = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=cfg.MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoded_text["input_ids"].to(device)
            attention_mask =encoded_text["attention_mask"].to(device)
            images = batch["image"].to(device)
            mass = batch["mass"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                mass=mass
            )

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * targets.size(0)
            total_train_samples += targets.size(0)


        avg_train_mae = total_train_loss / total_train_samples

        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in val_data_loader:
                texts = batch["text"]

                encoded_text = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=cfg.MAX_LENGTH,
                    return_tensors="pt"
                )

                input_ids = encoded_text["input_ids"].to(device)
                attention_mask = encoded_text["attention_mask"].to(device)
                images = batch["image"].to(device)
                mass = batch["mass"].to(device)
                targets = batch["target"].to(device)

                outputs=model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    mass=mass
                )

                loss = criterion(outputs, targets)
                total_val_loss += loss.item() * targets.size(0)
                total_val_samples += targets.size(0)

        avg_val_mae = total_val_loss / total_val_samples

        history["train_mae"].append(avg_train_mae)
        history["val_mae"].append(avg_val_mae)

        print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        print(f"Train MAE: {avg_train_mae:.2f}")
        print(f"Validation MAE: {avg_val_mae:.2f}")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            print("Best model saved")

    return history

def evaluate(cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    _, _, test_loader = get_dataloaders(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

    model = CalorieModel(cfg)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.L1Loss()

    total_test_loss = 0.0
    total_test_samples = 0
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["text"]

            encoded_text = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=cfg.MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoded_text["input_ids"].to(device)
            attention_mask = encoded_text["attention_mask"].to(device)
            images = batch["image"].to(device)
            mass = batch["mass"].to(device)
            targets = batch["target"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                mass=mass
            )

            loss = criterion(outputs, targets)

            total_test_loss += loss.item() * targets.size(0)
            total_test_samples += targets.size(0)

            for i in range(targets.size(0)):
                true_calories = targets[i].item()
                predicted_calories = outputs[i].item()

                predictions.append({
                    "dish_id": batch["dish_id"][i],
                    "ingredients": texts[i],
                    "mass": mass[i].item(),
                    "true_calories": true_calories,
                    "predicted_calories": predicted_calories,
                    "absolute_error": abs(true_calories - predicted_calories)
                })

    test_mae = total_test_loss / total_test_samples
    predictions_df = pd.DataFrame(predictions)

    print(f"Test MAE: {test_mae:.2f}")

    return test_mae, predictions_df





