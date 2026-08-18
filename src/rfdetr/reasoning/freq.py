# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frequency-domain patch encoder for the FFT consistency-reasoning plugin.

Extracts a compact spectral signature ``Ffreq`` from an image region by:

1. **Scale normalisation** — the patch is bilinearly resized to a fixed
   :attr:`FreqPatchEncoder.patch_size` *before* the FFT.  Large objects
   naturally contain more low-frequency energy and small objects more
   high-frequency energy; resizing decouples "area" from "frequency" so the
   spectrum encodes texture rather than box size.
2. **2D FFT magnitude spectrum** — a real 2D FFT (:func:`torch.fft.rfft2`)
   yields the complex spectrum; the magnitude ``|X|`` is the rotation-variant
   energy layout the plugin reasons over.
3. **Learnable frequency-domain filter** — a per-frequency learnable mask
   (initialised to ``1`` = identity) lets the network decide which frequency
   combinations matter most for pairwise reasoning.
4. **Compression** — the filtered spectrum is flattened and projected by a
   lightweight MLP into the low-dimensional ``Ffreq`` vector.

The module also exposes the normalised **high-frequency energy ratio**
(:meth:`FreqPatchEncoder.high_freq_ratio`) which the CRDe decoder uses as a
dynamic gate: a patch whose high-frequency content is severely missing is
treated as *extremely blurry* (reason by semantic co-occurrence), while a
patch that retains high frequencies is treated as *occluded-but-clear* (reason
by geometry).

The FFT, the learnable filter and the MLP all run in float32
(:func:`torch.fft.rfft2` rejects half/bfloat16, and the MLP weights are
float32).  When used inside a model cast to half/bf16 the module is kept in
float32 (or its parameters are cast alongside the input).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.models.math import MLP


class FreqPatchEncoder(nn.Module):
    """Encode an image patch into a compact spectral feature vector.

    Args:
        freq_dim: Output dimension of the spectral feature ``Ffreq``.
        patch_size: Fixed square size (in pixels) patches are resized to before
            the FFT.  Also the scale-normalisation target: all patches are
            bilinearly resized to ``patch_size x patch_size`` so area and
            frequency do not couple.
        hidden_dim: Hidden width of the compression MLP.
    """

    def __init__(self, freq_dim: int = 64, patch_size: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        # rfft2 of a (P, P) patch yields a (P, P // 2 + 1) complex spectrum; the
        # DC (mean-intensity) column is dropped, leaving P * (P // 2) entries.
        self.spectral_h = patch_size
        self.spectral_w = patch_size // 2
        # Learnable per-frequency filter, initialised to 1 (identity).
        self.freq_mask = nn.Parameter(torch.ones(1, 1, self.spectral_h, self.spectral_w))
        flat_dim = self.spectral_h * self.spectral_w
        self.proj = MLP(flat_dim, hidden_dim, freq_dim, 2)

    def forward(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        """Encode patches into spectral features.

        Args:
            patches: Image patches of shape ``(N, 3, H, W)`` with arbitrary
                ``H``/``W`` (they are resized internally), values in ``[0, 1]``.

        Returns:
            A tuple ``(ffreq, high_freq_ratio)``:
            - ``ffreq`` of shape ``(N, freq_dim)`` — the spectral feature vector.
            - ``high_freq_ratio`` of shape ``(N,)`` in ``[0, 1]`` — the fraction
              of spectral energy above the median frequency (dynamic gate input).
        """
        if patches.numel() == 0:
            # torch.fft.rfft2 rejects an empty batch with an MKL FFT error, and
            # F.interpolate of a zero-width batch is also unreliable, so short-
            # circuit before the FFT.
            empty = patches.new_empty((patches.shape[0], self.freq_dim))
            empty_ratio = patches.new_empty((patches.shape[0],))
            return empty, empty_ratio

        # Scale normalisation: fixed-size bilinear resize decouples area/frequency.
        patches = F.interpolate(
            patches,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        # The FFT and the whole spectral pipeline run in float32: rfft2 rejects
        # half/bfloat16, and the MLP weights are float32 too.  (When the module
        # is used inside a model that was cast to half/bf16, the caller keeps
        # this module in float32 or casts its parameters alongside the input.)
        spectrum = torch.fft.rfft2(patches.to(torch.float32), dim=(-2, -1))
        magnitude = spectrum.abs()  # (N, 3, P, P // 2 + 1)
        # Average over colour channels so the learnable mask is per-frequency.
        magnitude = magnitude.mean(dim=1, keepdim=True)  # (N, 1, P, P // 2 + 1)
        # DC component carries the mean intensity, not texture; drop it and
        # log-scale the remaining energy so the spectrum is well-conditioned.
        magnitude = magnitude[..., 1:]  # (N, 1, P, P // 2 - 1)
        magnitude = torch.log1p(magnitude)
        filtered = magnitude * self.freq_mask.to(magnitude.dtype)
        flat = filtered.flatten(1)
        ffreq = self.proj(flat)
        high_freq_ratio = self._high_freq_ratio(magnitude)
        return ffreq, high_freq_ratio

    def _high_freq_ratio(self, magnitude: Tensor) -> Tensor:
        """Normalised fraction of spectral energy above the median frequency.

        Args:
            magnitude: Log-scaled magnitude spectrum of shape ``(N, 1, H, W)``
                (DC component already removed).

        Returns:
            A tensor of shape ``(N,)`` in ``[0, 1]``.
        """
        energy = magnitude.squeeze(1)  # (N, H, W)
        total = energy.sum(dim=(-2, -1))  # (N,)
        # Row index = vertical frequency; take the top half as "high frequency".
        rows = energy.shape[-2]
        high = energy[..., rows // 2 :, :].sum(dim=(-2, -1))
        ratio = high / total.clamp_min(1e-6)
        return ratio
