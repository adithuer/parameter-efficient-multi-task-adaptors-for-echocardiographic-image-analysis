import math
import torch
import torch.nn as nn
from functorch.einops import rearrange
from torch import einsum
from typing import List, Tuple, Union


class PatchEmbeddingLayer(nn.Module):
    def __init__(
        self, dim_in: int, dim_emb: int, kernel_size: int, linear_emb: bool = False
    ) -> None:
        super(PatchEmbeddingLayer, self).__init__()
        self.linear_emb = linear_emb

        if self.linear_emb:
            self.patch_embedding = torch.nn.Linear(dim_in * kernel_size**2, dim_emb)
        else:
            self.patch_embedding = nn.Conv2d(
                dim_in * kernel_size**2, dim_emb, kernel_size=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear_emb:
            b, _, h, w = x.shape
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.patch_embedding(x)
            x = rearrange(x, "b (h w) c -> b c h w", b=b, h=h, w=w)
        else:
            x = self.patch_embedding(x)
        return x


class PatchLayer(nn.Module):
    def __init__(
        self, kernel_size: int, padding: int, stride: int, dilation: int = 1
    ) -> None:
        super(PatchLayer, self).__init__()

        self.patch_layer = nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_layer(x)

        b, c, n_patches = x.shape
        h = int(math.sqrt(n_patches))
        x = rearrange(x, "b c (h w) -> b c h w", b=b, c=c, h=h)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, reduction_ratio: int) -> None:
        super(EfficientSelfAttention, self).__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=reduction_ratio,
            stride=reduction_ratio,
            bias=False,
        )
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        heads = self.heads
        identity = x

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=heads), (q, k, v)
        )

        # q @ k.transpose(-2, -1) * self.scale
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # attn @ V
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) (x y) c -> b (h c) x y", h=heads, x=h, y=w)
        out = self.to_out(out)

        out = out + identity
        return out


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias=True,
    ) -> None:
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=dim_in,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MixFFN(nn.Module):
    def __init__(
        self, dim_in: int, expansion_factor: int, conv_only: bool = True
    ) -> None:
        super(MixFFN, self).__init__()
        self.conv_only = conv_only
        dim_hidden = dim_in * expansion_factor

        if self.conv_only:
            self.mix_ffn = nn.Sequential(
                nn.Conv2d(dim_in, dim_hidden, kernel_size=1, stride=1, padding=0),
                DepthwiseSeparableConv2d(
                    dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1
                ),
                nn.GELU(),
                nn.Conv2d(dim_hidden, dim_in, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.linear_down = nn.Linear(dim_in, dim_hidden)
            self.linear_up = nn.Linear(dim_hidden, dim_in)
            self.gelu = nn.GELU()
            self.conv = DepthwiseSeparableConv2d(
                dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.conv_only:
            x = self.mix_ffn(x)
        else:
            b, _, h, w = x.shape
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.linear_down(x)
            x = rearrange(x, "b (h w) c -> b c h w", b=b, h=h, w=w)
            x = self.conv(x)
            x = self.gelu(x)
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.linear_up(x)
            x = rearrange(x, "b (h w) c -> b c h w", b=b, h=h, w=w)

        x = x + identity

        return x


class MiTLayer(nn.Module):
    def __init__(
        self,
        dim_emb: int,
        n_heads: int,
        reduction_ratio: int,
        expansion_factor: int,
        conv_only: bool,
    ) -> None:
        super(MiTLayer, self).__init__()

        self.self_attn = EfficientSelfAttention(dim_emb, n_heads, reduction_ratio)
        self.mix_ffn = MixFFN(dim_emb, expansion_factor, conv_only)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        patch_emb_shape = patch_emb.shape[1:]

        out = nn.functional.layer_norm(patch_emb, patch_emb_shape)
        out = self.self_attn(patch_emb)

        out = nn.functional.layer_norm(patch_emb, patch_emb_shape)
        out = self.mix_ffn(out)

        return out


class SegformerStage(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        padding: int,
        stride: int,
        dim_in: int,
        dim_emb: int,
        n_heads: int,
        reduction_ratio: int,
        expansion_factor: int,
        n_layers: int,
        conv_only: bool = True,
        linear_emb: bool = False,
    ) -> None:
        super(SegformerStage, self).__init__()

        self.patch_layer = PatchLayer(kernel_size, padding, stride)
        self.patch_embedding = PatchEmbeddingLayer(
            dim_in, dim_emb, kernel_size, linear_emb
        )

        self.mit_layers = nn.Sequential()
        for n_layer in range(n_layers):
            self.mit_layers.add_module(
                f"MiTLayer_{n_layer}",
                MiTLayer(
                    dim_emb, n_heads, reduction_ratio, expansion_factor, conv_only
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = self.patch_layer(x)
        patch_emb = self.patch_embedding(patches)
        out = self.mit_layers(patch_emb)

        return out


class SegformerNeck(nn.Module):
    def __init__(self, dim_emb: List[int], dim_unified: int, n_stages: int, conv_only: bool = True) -> None:
        super(SegformerNeck, self).__init__()
        self.conv_only = conv_only

        self.unify = nn.ModuleList([])
        self.upsample = nn.ModuleList([])
        for i in range(n_stages):
            if self.conv_only:
                self.unify.append(nn.Conv2d(dim_emb[i], dim_unified, kernel_size=1))
            else:
                self.unify.append(nn.Linear(dim_emb[i], dim_unified))

            self.upsample.append(nn.UpsamplingBilinear2d(scale_factor=2**i))

    def forward(self, outputs: list) -> torch.Tensor:
        outputs_upsampled = []
        for i, output in enumerate(outputs):
            if self.conv_only:
                out = self.unify[i](output)
            else:
                b, _, h, w = output.shape
                out = rearrange(output, "b c h w -> b (h w) c")
                out = self.unify[i](out)
                out = rearrange(out, "b (h w) c -> b c h w", b=b, h=h, w=w)

            out = self.upsample[i](out)
            outputs_upsampled.append(out)

        out = torch.cat(outputs_upsampled, dim=1)
        return out


class SegmentationHead(nn.Module):
    def __init__(self, dim_unified: int, n_classes: int, img_size: Union[Tuple[int,int], int], conv_only: bool = True) -> None:
        super(SegmentationHead, self).__init__()
        self.conv_only = conv_only
        self.img_size = img_size

        if self.conv_only:
            self.fuse = nn.Sequential(
                nn.Conv2d(4 * dim_unified, dim_unified, kernel_size=1),
                nn.BatchNorm2d(dim_unified),
                nn.ReLU(),
            )
        else:
            self.fuse = nn.Linear(4 * dim_unified, dim_unified)
            self.batchnorm = nn.BatchNorm1d(dim_unified)
            self.relu = nn.ReLU()

        self.predict = nn.Conv2d(dim_unified, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_only:
            out = self.fuse(x)
        else:
            b, _, h, w = x.shape
            out = rearrange(x, "b c h w -> b (h w) c")
            out = self.fuse(out)
            out = self.batchnorm(out.permute(0, 2, 1))
            out = self.relu(out)
            out = rearrange(out, "b c (h w) -> b c h w", b=b, h=h, w=w)

        out = self.predict(out)

        out = nn.functional.interpolate(
            out, size=self.img_size, mode="bilinear", align_corners=False
        )

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dim_emb: List[int],
        reduction_ratio: List[int],
        n_heads: List[int],
        expansion_factor: List[int],
        n_layers: List[int],
        dim_in: int,
        n_stages: int,
        linear_emb: bool,
        conv_only: bool,
    ) -> None:
        super(Encoder, self).__init__()

        self.stages = nn.ModuleList([])
        for i in range(n_stages):
            self.stages.append(
                SegformerStage(
                    kernel_size=kernel_size[i],
                    padding=padding[i],
                    stride=stride[i],
                    dim_in=dim_in,
                    dim_emb=dim_emb[i],
                    n_heads=n_heads[i],
                    reduction_ratio=reduction_ratio[i],
                    expansion_factor=expansion_factor[i],
                    n_layers=n_layers[i],
                    conv_only=conv_only,
                    linear_emb=linear_emb,
                )
            )

            dim_in = dim_emb[i]

    def forward(self, x: torch.Tensor) -> list:
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)

        return outputs


class Segformer(nn.Module):
    def __init__(
        self,
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dim_emb: List[int],
        reduction_ratio: List[int],
        n_heads: List[int],
        expansion_factor: List[int],
        n_layers: List[int],
        dim_unified: int,
        n_classes: int,
        img_size: Tuple[int],
        dim_in: int = 3,
        n_stages: int = 4,
        linear_emb: bool = False,
        conv_only: bool = True,
    ) -> None:
        super(Segformer, self).__init__()

        self.encoder = Encoder(kernel_size, stride, padding, dim_emb, reduction_ratio, n_heads,
                               expansion_factor, n_layers, dim_in, n_stages, linear_emb, conv_only)
        self.neck = SegformerNeck(dim_emb, dim_unified, n_stages, conv_only)
        self.head = SegmentationHead(dim_unified, n_classes, img_size, conv_only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x)
        out = self.neck(outputs)
        out = self.head(out)

        return out
