# src/models/classification/multimodal_classifier.py
import torch
import torch.nn as nn
from ..fusion.multimodal_transformer import MultimodalTransformer


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, text_dim=768, image_dim=2048,
                 audio_dim=128, video_dim=512, embed_dim=512):
        super().__init__()

        # Multimodal fusion model
        self.fusion_model = MultimodalTransformer(
            text_dim=text_dim,
            image_dim=image_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            embed_dim=embed_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, text_features=None, image_features=None,
                audio_features=None, video_features=None):
        # Get multimodal embedding
        fusion_embedding = self.fusion_model(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features,
            video_features=video_features
        )

        # Classify
        logits = self.classifier(fusion_embedding)

        return logits

    def predict(self, text_features=None, image_features=None,
                audio_features=None, video_features=None):
        logits = self.forward(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features,
            video_features=video_features
        )

        # Get predicted class
        _, predicted = torch.max(logits, 1)

        return predicted