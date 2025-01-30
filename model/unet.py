import torch
import torch.nn as nn
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, dim_in: int, out_dim: int) -> None:
        super(DoubleConv, self).__init__()

        self.doubleConv = nn.Sequential(
            nn.Conv2d(dim_in, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.doubleConv(x)
        return x


class Up(nn.Module):
    def __init__(self, dim_in: int, out_dim: int) -> None:
        super(Up, self).__init__()

        self.upConv = nn.ConvTranspose2d(dim_in, out_dim, kernel_size=(2, 2), stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upConv(x)
        return x


class Down(nn.Module):
    def __init__(self) -> None:
        super(Down, self).__init__()

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, dim_in: int, out_dim: int) -> None:
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Conv2d(dim_in, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_base: int) -> None:
        super(Encoder, self).__init__()

        self.doubleConv1 = DoubleConv(dim_in, dim_base)
        self.doubleConv2 = DoubleConv(dim_base, dim_base * 2)
        self.doubleConv3 = DoubleConv(dim_base * 2, dim_base * 4)
        self.doubleConv4 = DoubleConv(dim_base * 4, dim_base * 8)
        self.doubleConv5 = DoubleConv(dim_base * 8, dim_base * 16)

        self.down = Down()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        x = self.doubleConv1(x)
        skips.append(x)
        x = self.down(x)

        x = self.doubleConv2(x)
        skips.append(x)
        x = self.down(x)

        x = self.doubleConv3(x)
        skips.append(x)
        x = self.down(x)

        x = self.doubleConv4(x)
        skips.append(x)
        x = self.down(x)

        x = self.doubleConv5(x)

        return x, skips


class Decoder(nn.Module):
    def __init__(self, dim_base: int) -> None:
        super(Decoder, self).__init__()

        self.doubleConv6 = DoubleConv(dim_base * 16, dim_base * 8)
        self.doubleConv7 = DoubleConv(dim_base * 8, dim_base * 4)
        self.doubleConv8 = DoubleConv(dim_base * 4, dim_base * 2)
        self.doubleConv9 = DoubleConv(dim_base * 2, dim_base)

        self.upConv1 = Up(dim_base * 16, dim_base * 8)
        self.upConv2 = Up(dim_base * 8, dim_base * 4)
        self.upConv3 = Up(dim_base * 4, dim_base * 2)
        self.upConv4 = Up(dim_base * 2, dim_base)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x1, x2, x3, x4 = skips

        x = self.upConv1(x)

        x = self.cropAndConcat(x, x4)
        x = self.doubleConv6(x)
        x = self.upConv2(x)

        x = self.cropAndConcat(x, x3)
        x = self.doubleConv7(x)
        x = self.upConv3(x)

        x = self.cropAndConcat(x, x2)
        x = self.doubleConv8(x)
        x = self.upConv4(x)

        x = self.cropAndConcat(x, x1)
        x = self.doubleConv9(x)

        return x

    def cropAndConcat(self, x_encoded: torch.Tensor, x_decoded: torch.Tensor) -> torch.tensor:
        """crop x to the decoder x shape"""
        x_encoded = torchvision.transforms.functional.center_crop(
            x_encoded, [x_decoded.shape[2], x_decoded.shape[3]]
        )
        return torch.cat([x_decoded, x_encoded], dim=1)


class UNet(nn.Module):
    def __init__(self, *, dim_in: int, dim_base: int, n_classes: int) -> None:
        super(UNet, self).__init__()

        self.encoder = Encoder(dim_in, dim_base)
        self.decoder = Decoder(dim_base)
        self.bottleneck = Bottleneck(dim_base, n_classes)

    def forward(self, x: torch.Tensor, threshold=None) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.bottleneck(x)

        if not self.training and threshold:
            x = torch.sigmoid(x)
            thresh = torch.tensor([threshold])
            x = (x >= thresh).float()

        return x
