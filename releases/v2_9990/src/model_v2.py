import torch
from torch import nn


class Fixed6DigitCRNN(nn.Module):
    def __init__(self, hidden_size: int = 192, num_layers: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, 24))
        self.sequence_model = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        self.position_pool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 2, 10),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        sequence = self.sequence_pool(features).squeeze(2).permute(0, 2, 1).contiguous()
        sequence, _ = self.sequence_model(sequence)
        positions = self.position_pool(sequence.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return self.classifier(positions)
