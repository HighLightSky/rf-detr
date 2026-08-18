# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pairwise spatial position encoding (CTRP formula-10 port).

The original CTRP ``compute_pairwise_encodings_with_normalized`` builds a
32-dimensional per-pair spatial encoding (mode ``bbox_with_relative``):

- 4 dims of normalised box-centre coordinates for both boxes,
- 4 dims of normalised box width/height for both boxes,
- 4 dims of box angles ``sin(theta1), cos(theta1), sin(theta2), cos(theta2)``,
- 4 dims of relative centre distance and direction,
- followed by ``log`` of the first 12 dims (log is skipped for the angle terms
  because it produced NaNs).

RF-DETR detects **horizontal** boxes (no rotation), so the 4 angle dims are
constant ``(0, 1, 0, 1)`` and carry no information.  The FFT plugin **replaces
those 4 angle dims with a frequency-derived 4-vector** (:meth:`build_pairwise_pe`'s
``freq_diff_4d``) so the position encoding explicitly encodes "how texturally
different the two targets are" — a frequency-domain prior for spatial reasoning.
The remaining 12 geometric dims and their log copy are preserved, keeping the
output at 32 dims to match the original ``pairwise_head`` input size.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-3


def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000.0) -> Tensor:
    """Sinusoidal positional embedding for 2D points (``compute_sinusoidal_pe`` port).

    Args:
        pos_tensor: Normalised coordinates of shape ``(N, 2)`` (x, y in ``[0, 1]``).
        temperature: Temperature in the sinusoidal functions.

    Returns:
        Embedding of shape ``(N, 256)``: 128 dims for y then 128 dims for x.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, 0] * scale
    y_embed = pos_tensor[:, 1] * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    return torch.cat((pos_y, pos_x), dim=1)


def _xyxy_to_cxcywh_relative(boxes: Tensor, img_shape: tuple[int, int] | Tensor) -> Tensor:
    """Convert xyxy pixel boxes to normalised cxcywh.

    Args:
        boxes: Pixel boxes of shape ``(N, 4)`` as ``[x1, y1, x2, y2]``.
        img_shape: Either an image ``(height, width)`` tuple (all boxes share it)
            or a ``(N, 2)`` tensor of per-box image heights/widths.

    Returns:
        Normalised ``(cx, cy, w, h)`` of shape ``(N, 4)``.
    """
    if isinstance(img_shape, tuple):
        h_t: Tensor = torch.as_tensor(img_shape[0], dtype=boxes.dtype, device=boxes.device)
        w_t: Tensor = torch.as_tensor(img_shape[1], dtype=boxes.dtype, device=boxes.device)
    else:
        h_t = img_shape[:, 0]
        w_t = img_shape[:, 1]
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0 / w_t
    cy = (y1 + y2) / 2.0 / h_t
    bw = (x2 - x1) / w_t
    bh = (y2 - y1) / h_t
    return torch.stack([cx, cy, bw, bh], dim=-1)


def build_pairwise_pe(
    boxes_a: Tensor,
    boxes_b: Tensor,
    freq_diff_4d: Tensor,
    img_shape: tuple[int, int] | Tensor,
) -> Tensor:
    """Build the 32-dim pairwise spatial encoding with frequency-replaced angles.

    Args:
        boxes_a: ``(N, 4)`` pixel boxes of the clear target A as ``[x1, y1, x2, y2]``.
        boxes_b: ``(N, 4)`` pixel boxes of the blurry target B as ``[x1, y1, x2, y2]``.
        freq_diff_4d: ``(N, 4)`` frequency-derived vector that replaces the four
            angle dims of the original encoding.
        img_shape: Either an image ``(height, width)`` tuple or a ``(N, 2)``
            tensor of per-box image heights/widths used to normalise coordinates.

    Returns:
        Pairwise encoding of shape ``(N, 32)``.
    """
    a = _xyxy_to_cxcywh_relative(boxes_a, img_shape)
    b = _xyxy_to_cxcywh_relative(boxes_b, img_shape)

    c1_x, c1_y = a[:, 0], a[:, 1]
    c2_x, c2_y = b[:, 0], b[:, 1]
    b1_w, b1_h = a[:, 2], a[:, 3]
    b2_w, b2_h = b[:, 2], b[:, 3]

    d_x = torch.abs(c2_x - c1_x) / (b1_w + _EPS)
    d_y = torch.abs(c2_y - c1_y) / (b1_h + _EPS)

    # 16-dim raw encoding matching the original layout: 8 geometric dims, then
    # the 4 angle dims (indices 8:12) replaced by freq_diff_4d, then 4 dims of
    # relative centre distance/direction.
    geometric = torch.stack(
        [
            # Relative position of box centre (4)
            c1_x,
            c1_y,
            c2_x,
            c2_y,
            # Relative box width and height (4)
            b1_w,
            b1_h,
            b2_w,
            b2_h,
        ],
        dim=-1,
    )
    freq = freq_diff_4d.to(geometric.dtype).clamp_min(_EPS)  # (N, 4)
    relative = torch.stack(
        [
            # Relative distance and direction of B w.r.t. A (4)
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ],
        dim=-1,
    )
    raw = torch.cat([geometric, freq, relative], dim=-1)  # (N, 16)

    # 16 dims + log copy of the same 16 = 32.  Degenerate boxes (x1 > x2 or a
    # negative width from unvalidated input) would make ``raw`` negative and
    # ``log`` produce NaN, so clamp before taking the log — the original code
    # applied the same ``f[f < 0] = eps`` guard.
    safe = raw.clamp_min(_EPS)
    return torch.cat([raw, torch.log(safe)], dim=-1)
