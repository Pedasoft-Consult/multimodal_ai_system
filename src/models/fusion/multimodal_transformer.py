# src/models/fusion/multimodal_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return self.layer_norm(x + attn_output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        ff_output = self.ff(x)
        return self.layer_norm(x + ff_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class ModalityEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.projection(x)


class MultimodalTransformer(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, audio_dim=128, video_dim=512,
                 embed_dim=512, num_heads=8, ff_dim=2048, num_layers=4):
        super().__init__()

        # Modality-specific embeddings
        self.text_embedding = ModalityEmbedding(text_dim, embed_dim)
        self.image_embedding = ModalityEmbedding(image_dim, embed_dim)
        self.audio_embedding = ModalityEmbedding(audio_dim, embed_dim)
        self.video_embedding = ModalityEmbedding(video_dim, embed_dim)

        # Modality type embeddings (to distinguish different modalities)
        self.modality_type_embedding = nn.Embedding(4, embed_dim)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_features=None, image_features=None,
                audio_features=None, video_features=None):
        # Process each modality if available
        embeddings = []

        if text_features is not None:
            text_emb = self.text_embedding(text_features)
            # Add modality type embedding (0 for text)
            text_emb = text_emb + self.modality_type_embedding(torch.zeros(text_emb.size(0), dtype=torch.long))
            embeddings.append(text_emb)

        if image_features is not None:
            image_emb = self.image_embedding(image_features)
            # Add modality type embedding (1 for image)
            image_emb = image_emb + self.modality_type_embedding(torch.ones(image_emb.size(0), dtype=torch.long))
            embeddings.append(image_emb)

        if audio_features is not None:
            audio_emb = self.audio_embedding(audio_features)
            # Add modality type embedding (2 for audio)
            audio_emb = audio_emb + self.modality_type_embedding(2 * torch.ones(audio_emb.size(0), dtype=torch.long))
            embeddings.append(audio_emb)

        if video_features is not None:
            video_emb = self.video_embedding(video_features)
            # Add modality type embedding (3 for video)
            video_emb = video_emb + self.modality_type_embedding(3 * torch.ones(video_emb.size(0), dtype=torch.long))
            embeddings.append(video_emb)

        # Combine all modalities
        if len(embeddings) == 0:
            raise ValueError("At least one modality must be provided")

        # Stack embeddings along sequence dimension
        x = torch.stack(embeddings, dim=1)

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Use mean pooling for final representation
        x = torch.mean(x, dim=1)

        # Final output projection
        output = self.output_layer(x)

        return output