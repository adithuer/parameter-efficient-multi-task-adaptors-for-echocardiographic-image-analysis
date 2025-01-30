import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, dim_in: int, img_size: int) -> None:
        super(Regressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Conv2d(dim_in, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(128 * (img_size // 8) ** 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regressor(x)
        return x