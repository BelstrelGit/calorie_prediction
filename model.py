import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class CalorieModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(cfg.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)

        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.num_features

        self.text_projection = nn.Linear(text_dim,256)
        self.image_projection = nn.Linear(image_dim, 256)

        self.regression_head = nn.Sequential(
            nn.Linear(256 + 256 +1, 256), #text_features[batch,256], image_features[batch,256],mass[batch,1]
            nn.ReLU(), #Rectified Linear Unit (выпрямленный линейный модуль->  если x < 0 → 0 , если x > 0 → x
            nn.Dropout(0.2), #регуляризация
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, images, mass):
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)

        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features)

        features = torch.cat([text_features, image_features, mass],dim=1)
        return  self.regression_head(features)

