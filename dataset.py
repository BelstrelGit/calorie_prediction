import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


def split_data(dish_df, val_size=0.15, random_state=42):
    train_df = dish_df[dish_df["split"] == "train"].copy()
    test_df = dish_df[dish_df["split"] == "test"].copy()
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df

def decode_ingredients(ingredients, ingredients_dict):
    ingredient_ids = ingredients.split(";")
    ingredient_names = [ingredients_dict[ingredient_id] for ingredient_id in ingredient_ids]
    return ", ".join(ingredient_names)


def load_data(data_dir):
    dish_path = os.path.join(data_dir, "dish.csv")
    ingredients_path = os.path.join(data_dir, "ingredients.csv")
    dish_df = pd.read_csv(dish_path)
    ingredients_df = pd.read_csv(ingredients_path)
    ingredients_df["ingredient_id"] = ingredients_df["id"].apply(lambda x: f"ingr_{x:010d}")
    ingredients_dict = dict(zip(ingredients_df["ingredient_id"], ingredients_df["ingr"]))
    dish_df["ingredients_text"] = dish_df["ingredients"].apply(lambda x: decode_ingredients(x, ingredients_dict))
    return dish_df


def get_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2() #PyTorch tensor [3, H, W]
    ])

def get_dataloaders(cfg):
    dish_df = load_data(cfg.DATA_DIR)
    train_df, val_df, test_df = split_data(dish_df, cfg.VAL_SIZE, cfg.SEED)
    if cfg.DEBUG:
        train_df = train_df.sample(cfg.DEBUG_TRAIN_SIZE, random_state=cfg.SEED)
        val_df = val_df.sample(cfg.DEBUG_VAL_SIZE, random_state=cfg.SEED)

    transform = get_transforms(cfg.IMAGE_SIZE)
    train_dataset = FoodDataset(train_df, cfg.DATA_DIR, transform)
    val_dataset = FoodDataset(val_df, cfg.DATA_DIR, transform)
    test_dataset = FoodDataset(test_df, cfg.DATA_DIR, transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

class FoodDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row["dish_id"]
        img_path = os.path.join(self.data_dir, "images", dish_id,"rgb.png")
        image  = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        text = row["ingredients_text"]
        mass = np.array([row["total_mass"]], dtype=np.float32)
        target = np.array([row["total_calories"]], dtype = np.float32)
        return {
            "image": image,
            "text": text,
            "mass": mass,
            "target": target,
            "dish_id": dish_id
        }
