import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, img_size: int) -> None:
        super(ClassifierHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(dim_in * (img_size ** 2), 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim_out),
        )

    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        x = self.head(x)
        
        if not self.training and threshold:
            x = nn.functional.softmax(x, dim=-1)
            x = (x >= threshold).float()

        return x