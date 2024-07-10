# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/backbones/vision_transformer.py."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

import torch
from timm.layers import (
    LayerType,
    Mlp,
    PatchDropout,
    SwiGLUPacked,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    resample_patch_embed,
    trunc_normal_,
)
from timm.layers import PatchEmbed as TimmPatchEmbed
from timm.models._manipulate import adapt_input_conv
from timm.models.vision_transformer import Block
from torch import nn

from otx.algo.modules.base_module import BaseModule

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


VIT_ARCH_TYPE = Literal[
    "vit-t",
    "vit-tiny",
    "vit-s",
    "vit-small",
    "vit-b",
    "vit-base",
    "vit-l",
    "vit-large",
    "vit-h",
    "vit-huge",
    "dinov2-s",
    "dinov2-small",
    "dinov2-b",
    "dinov2-base",
    "dinov2-l",
    "dinov2-large",
    "dinov2-g",
    "dinov2-giant",
]


class VisionTransformer(BaseModule):
    """Implementation of Vision Transformer from Timm.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
        - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    Args:
        arch: Vision Transformer architecture.
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Mumber of classes for classification head.
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values (layer-scale enabled if not None).
        class_token: Use class token.
        no_embed_class: Don't include position embeddings for class (or reg) tokens.
        reg_tokens: Number of register tokens.
        drop_rate: Head dropout rate.
        pos_drop_rate: Position embedding dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        weight_init: Weight initialization scheme.
        fix_init: Apply weight initialization fix (scaling w/ layer index).
        embed_layer: Patch embedding layer.
        norm_layer: Normalization layer.
        act_layer: MLP activation layer.
        block_fn: Transformer block layer.
    """

    arch_zoo = {  # noqa: RUF012
        **dict.fromkeys(
            ["vit-t", "vit-tiny"],
            {
                "patch_size": 16,
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
            },
        ),
        **dict.fromkeys(
            ["vit-s", "vit-small"],
            {
                "patch_size": 16,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
            },
        ),
        **dict.fromkeys(
            ["vit-b", "vit-base"],
            {
                "patch_size": 16,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
            },
        ),
        **dict.fromkeys(
            ["vit-l", "vit-large"],
            {
                "patch_size": 16,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
            },
        ),
        **dict.fromkeys(
            ["vit-h", "vit-huge"],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                "patch_size": 16,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
            },
        ),
        **dict.fromkeys(
            ["dinov2-s", "dinov2-small"],
            {
                "patch_size": 14,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
            },
        ),
        **dict.fromkeys(
            ["dinov2-b", "dinov2-base"],
            {
                "patch_size": 14,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
            },
        ),
        **dict.fromkeys(
            ["dinov2-l", "dinov2-large"],
            {
                "patch_size": 14,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
            },
        ),
        **dict.fromkeys(
            ["dinov2-g", "dinov2-giant"],
            {
                "patch_size": 14,
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "mlp_ratio": 2.66667 * 2,
                "mlp_layer": SwiGLUPacked,
                "act_layer": nn.SiLU,
            },
        ),
    }

    def __init__(  # noqa: PLR0913
        self,
        arch: VIT_ARCH_TYPE = "vit-base",
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] | None = None,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = TimmPatchEmbed,
        block_fn: nn.Module = Block,
        mlp_layer: nn.Module | None = None,
        act_layer: LayerType | None = None,
        norm_layer: LayerType | None = None,
    ) -> None:
        super().__init__()
        if isinstance(arch, str):
            if arch not in set(self.arch_zoo):
                msg = f"Arch {arch} is not in default archs {set(self.arch_zoo)}"
                raise ValueError(msg)
            arch_settings = self.arch_zoo[arch]

        patch_size = patch_size or arch_settings["patch_size"]
        embed_dim = embed_dim or arch_settings["embed_dim"]
        depth = depth or arch_settings["depth"]
        num_heads = num_heads or arch_settings["num_heads"]
        mlp_layer = mlp_layer or arch_settings.get("mlp_layer", None) or Mlp
        mlp_ratio = mlp_ratio or arch_settings.get("mlp_ratio", None) or 4.0
        norm_layer = (
            get_norm_layer(norm_layer) or arch_settings.get("norm_layer", None) or partial(nn.LayerNorm, eps=1e-6)
        )
        act_layer = get_act_layer(act_layer) or arch_settings.get("act_layer", None) or nn.GELU

        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update({"strict_img_size": False, "output_fmt": "NHWC"})
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, embed_dim))

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ],
        )

        self.norm = norm_layer(embed_dim)

    def init_weights(self) -> None:
        """Initializes the weights of the VisionTransformer."""
        super().init_weights()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: Path, prefix: str = "") -> None:
        """Loads the pretrained weight to the VisionTransformer."""
        checkpoint_ext = checkpoint_path.suffix
        if checkpoint_ext == ".npz":
            _load_npz_weights(self, checkpoint_path, prefix)
        elif checkpoint_ext == ".pth":
            self.load_state_dict(torch.load(checkpoint_path), strict=False)
        else:
            msg = f"Unsupported `checkpoint_extension` {checkpoint_ext}, please choose from 'npz' or 'pth'."
            raise ValueError(msg)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Implements positional embedding."""
        if self.dynamic_img_size:
            b, h, w, c = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (h, w),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(b, -1, c)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
            x = x + pos_embed

        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        out_type: Literal["raw", "cls_token", "featmap", "avg_featmap"] = "cls_token",
    ) -> tuple:
        """Forward pass of the VisionTransformer model."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        if out_type == "raw":
            return (x,)
        if out_type == "cls_token":
            return (x[:, 0],)
        msg = f"Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}"
        raise ValueError(msg)

    def forward_explain(self, x: torch.Tensor) -> tuple:
        """Forward pass of the VisionTransformer model."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        if out_type == "raw":
            return (x,)
        if out_type == "cls_token":
            return (x[:, 0],)
        msg = f"Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}"
        raise ValueError(msg)


@torch.no_grad()
def _load_npz_weights(  # noqa: C901
    model: VisionTransformer,
    checkpoint_path: str,
    prefix: str = "",
) -> None:
    """Load weights from .npz checkpoints for official Google Brain Flax implementation."""
    import numpy as np

    def _n2p(w: np.ndarray, t: bool = True, idx: int | None = None) -> torch.Tensor:
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = "bilinear"
    antialias = False
    big_vision = False
    if not prefix:
        if "opt/target/embedding/kernel" in w:
            prefix = "opt/target/"
        elif "params/embedding/kernel" in w:
            prefix = "params/"
            big_vision = True
        elif "params/img/embedding/kernel" in w:
            prefix = "params/img/"
            big_vision = True

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])))
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(_n2p(w[f"{bp}conv{r + 1}/kernel"]))
                        getattr(block, f"norm{r + 1}").weight.copy_(_n2p(w[f"{bp}gn{r + 1}/scale"]))
                        getattr(block, f"norm{r + 1}").bias.copy_(_n2p(w[f"{bp}gn{r + 1}/bias"]))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f"{bp}conv_proj/kernel"]))
                        block.downsample.norm.weight.copy_(_n2p(w[f"{bp}gn_proj/scale"]))
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"]))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f"{prefix}pos_embedding"], t=False)
    else:
        pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
            block_prefix = f"{prefix}Transformer/encoderblock/"
            idx = i
        else:
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            idx = None
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"], idx=idx))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T for n in ("query", "key", "value")],
            ),
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1) for n in ("query", "key", "value")],
            ),
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"], idx=idx).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"], idx=idx))
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/scale"], idx=idx))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/bias"], idx=idx))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel"], idx=idx),
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias"], idx=idx),
            )
