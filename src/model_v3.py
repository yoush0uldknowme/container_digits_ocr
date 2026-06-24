import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class Fixed6DigitAttentionCRNN(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 48),
            ResidualBlock(48, 96),
            ResidualBlock(96, 192),
            ResidualBlock(192, 320),
        )
        self.height_pool = nn.AdaptiveAvgPool2d((1, 24))
        self.sequence_model = nn.LSTM(
            input_size=320,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        embedding_size = hidden_size * 2
        self.position_queries = nn.Parameter(torch.randn(1, 6, embedding_size) * 0.02)
        self.position_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Dropout(0.15),
            nn.Linear(embedding_size, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        sequence = self.height_pool(features).squeeze(2).permute(0, 2, 1).contiguous()
        sequence, _ = self.sequence_model(sequence)
        queries = self.position_queries.expand(images.size(0), -1, -1)
        positions, _ = self.position_attention(queries, sequence, sequence, need_weights=False)
        return self.classifier(positions)
