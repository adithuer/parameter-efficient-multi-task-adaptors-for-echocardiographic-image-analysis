import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, img_size) -> None:
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(dim_in, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(128 * (img_size // 8) ** 2, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim_out)
        )

    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        x = self.classifier(x)
        
        if not self.training and threshold:
            x = nn.functional.softmax(x, dim=-1)
            x = (x >= threshold).float()

        return x