# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Consistency-reasoning decoder (CRDe) for the FFT plugin.

A transformer decoder that re-scores a *blurry* candidate detection ``B`` given
one or more *clear* reference detections ``A``.  It is a light port of the
original CTRP ``SrtMaskDecoderLayer`` (mmrotate) with three FFT-driven changes:

- **Relation feature.**  The original fed RoI features of the union region into
  ``rel_feat_head``.  Here the relation feature is ``concat(Ffreq_A, Ffreq_B)``
  — the spectral signatures of the two patches — which already encode texture.
- **Gate injection.**  A learnable gate ``G = sigmoid(MLP(concat(Ffreq_A, Ffreq_B)))``
  is multiplied onto the cross-attention weight matrix.  ``G -> 1`` (A and B
  textures look similar) lets features propagate freely and strongly recalls
  missed objects; ``G -> 0`` (textures are very different, e.g. ground vs
  tire) suppresses propagation of that relation pair, curbing false alarms.
- **Dynamic semantic/geometric routing.**  The blurry patch's high-frequency
  energy ratio ``h`` routes between the two priors: when ``h`` is low (the
  target is *extremely* blurry) reasoning leans on semantic co-occurrence
  (guess from the subject's class); when ``h`` is high (occluded but clear) it
  leans on geometric position reasoning (the pairwise spatial encoding).

The decoder still uses the original structure: a self-attention block over the
relation queries (with a centre position embedding), then a cross-attention
block where each relation query attends to the model's class-embedding matrix
(as semantic key/value) with a spatial-prior fusion, then an FFN.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.reasoning.pe import build_pairwise_pe, compute_sinusoidal_pe


def _attention_weights(
    q: Tensor,
    k: Tensor,
    num_heads: int,
    scale: float,
    gate: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention weights with optional per-query gate.

    Args:
        q: Query tensor ``(N, d)`` with positional embedding already added.
        k: Key tensor ``(M, d)`` with positional embedding already added.
        num_heads: Number of attention heads (kept for interface parity; the
            small decoder computes a single-head softmax over the full width).
        scale: ``1 / sqrt(d / num_heads)``.
        gate: Optional per-query gate ``(N, 1)`` in ``[0, 1]`` multiplied onto
            the attention weights (no re-normalisation — the gate deliberately
            scales the propagated feature magnitude, suppressing propagation
            when it is near ``0``).

    Returns:
        Attention weights of shape ``(N, M)``.
    """
    # q: (N, d), k: (M, d)
    attn = (q @ k.transpose(-2, -1)) * scale  # (N, M)
    attn = attn.softmax(dim=-1)
    if gate is not None:
        attn = attn * gate  # (N, M)
    return attn


class MultiModalFusion(nn.Module):
    """Fuse the semantic and spatial priors with a frequency-driven blend.

    Ported from the original ``MultiModalFusion`` in ``srt_mask_decoder_layer.py``:
    two ``nn.Linear`` branches (one per modality), LayerNorm each, ReLU on the
    concatenation, then a small MLP.  The additional ``blend`` input weights the
    spatial branch by ``alpha`` and the semantic branch by ``1 - alpha`` where
    ``alpha`` is the high-frequency energy ratio of the blurry patch.
    """

    def __init__(self, sem_size: int, spa_size: int, out_size: int) -> None:
        """Initialise the fusion branches.

        Args:
            sem_size: Semantic-branch input width.
            spa_size: Spatial-branch input width.
            out_size: Output width.
        """
        super().__init__()
        self.sem_fc = nn.Linear(sem_size, out_size)
        self.spa_fc = nn.Linear(spa_size, out_size)
        self.ln_sem = nn.LayerNorm(out_size)
        self.ln_spa = nn.LayerNorm(out_size)
        sizes = [2 * out_size, int(out_size * 1.5), out_size]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, semantic: Tensor, spatial: Tensor, alpha: Tensor) -> Tensor:
        """Fuse semantic and spatial features.

        Args:
            semantic: Semantic features ``(N, sem_size)``.
            spatial: Spatial features ``(N, spa_size)``.
            alpha: Per-sample blend ``(N,)`` in ``[0, 1]`` (high-freq ratio of
                the blurry patch).  Higher alpha favours the spatial branch.

        Returns:
            Fused features ``(N, out_size)``.
        """
        x = (1 - alpha).unsqueeze(-1) * self.ln_sem(F.relu(self.sem_fc(semantic)))
        y = alpha.unsqueeze(-1) * self.ln_spa(F.relu(self.spa_fc(spatial)))
        z = F.relu(torch.cat([x, y], dim=-1))
        return cast(Tensor, self.mlp(z))


class ConsistencyDecoderLayer(nn.Module):
    """One CRDe layer: self-attention + gate-injected cross-attention + FFN."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        pairwise_encoding_dim: int,
        num_heads: int,
        ffn_interm_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the layer.

        Args:
            q_dim: Relation-query width (the Ffreq-derived feature width).
            kv_dim: Class-embedding width (rf-detr ``hidden_dim``, 256).
            pairwise_encoding_dim: Input width of the pairwise spatial encoding
                (32 for the ported formula-10 encoding).
            num_heads: Number of attention heads.
            ffn_interm_dim: FFN intermediate width.
            dropout: Dropout probability.
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        # The attention is computed as a single full-width softmax (see
        # ``_attention_weights``), so the correct scale is 1/sqrt(d) — not the
        # per-head dimension.  num_heads is kept for interface parity.
        self.scale = q_dim**-0.5

        # Self-attention over relation queries.
        self.attn_q_proj = nn.Linear(q_dim, q_dim)
        self.attn_k_proj = nn.Linear(q_dim, q_dim)
        self.attn_v_proj = nn.Linear(q_dim, q_dim)
        self.attn_qpos_proj = nn.Linear(256, q_dim)
        self.attn_kpos_proj = nn.Linear(256, q_dim)

        # Cross-attention: relation query attends to the class-embedding matrix.
        # With the semantics prior the query is concat(category_embedding, query).
        eq_dim = q_dim + kv_dim
        self.cross_q_proj = nn.Linear(eq_dim, q_dim)
        self.cross_k_proj = nn.Linear(kv_dim, q_dim)
        self.cross_v_proj = nn.Linear(kv_dim, q_dim)
        self.cross_qpos_proj = nn.Linear(256, q_dim)
        self.cross_kpos_proj = nn.Linear(kv_dim, q_dim)

        self.pairwise_head = nn.Sequential(
            nn.Linear(pairwise_encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, q_dim),
            nn.ReLU(),
        )
        self.mmf = MultiModalFusion(eq_dim, q_dim, eq_dim)

        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim),
        )
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        features: Tensor,
        category_embedding: Tensor,
        spatial: dict[str, Tensor],
        gate: Tensor,
        high_freq_ratio: Tensor,
    ) -> Tensor:
        """Run one decoder layer.

        Args:
            queries: Relation queries ``(N, q_dim)``.
            features: Class-embedding matrix ``(num_classes, kv_dim)`` used as
                cross-attention keys/values.
            category_embedding: Subject class embedding ``(N, kv_dim)``.
            spatial: Spatial prior dict with ``centre_pe`` ``(N, 256)`` and
                ``pairwise_feat`` ``(N, 32)``.
            gate: Per-query propagation gate ``(N, 1)`` in ``[0, 1]``.
            high_freq_ratio: Per-query high-frequency ratio ``(N,)`` in ``[0, 1]``
                used to blend semantic vs geometric priors.

        Returns:
            Refined queries ``(N, q_dim)``.
        """
        # ── self-attention over relation queries ───────────────────────────
        q = self.attn_q_proj(queries)
        k = self.attn_k_proj(queries)
        v = self.attn_v_proj(queries)
        q_p = self.attn_qpos_proj(spatial["centre_pe"])
        k_p = self.attn_kpos_proj(spatial["centre_pe"])
        q = q + q_p
        k = k + k_p
        attn = _attention_weights(q, k, self.num_heads, self.scale)
        out = attn @ v
        queries = self.ln1(queries + self.dropout1(out))

        # ── cross-attention: query attends to class-embedding matrix ───────
        # Semantic prior: the subject's class embedding concatenated to the query.
        explicit = torch.cat([category_embedding, queries], dim=-1)  # (N, q_dim+kv)
        # Spatial prior: pairwise encoding projected to q_dim.
        pairwise = self.pairwise_head(spatial["pairwise_feat"])  # (N, q_dim)
        # Dynamic routing: blurry (low high-freq) leans semantic, clear leans
        # geometric — the MMF blend ``alpha`` is the high-frequency ratio.
        explicit = self.mmf(explicit, pairwise, high_freq_ratio)

        q = self.cross_q_proj(explicit)
        k = self.cross_k_proj(features)
        v = self.cross_v_proj(features)
        q_p = self.cross_qpos_proj(spatial["centre_pe"])
        k_p = self.cross_kpos_proj(features)
        q = q + q_p
        k = k + k_p
        # Gate is multiplied onto the attention weights: G->1 propagates
        # (recall boost), G->0 suppresses (false-alarm control).
        attn = _attention_weights(q, k, self.num_heads, self.scale, gate=gate)
        out = attn @ v
        queries = self.ln2(queries + self.dropout2(out))
        queries = self.ln3(queries + self.dropout3(self.ffn(queries)))
        return queries


class ConsistencyDecoder(nn.Module):
    """Stack of :class:`ConsistencyDecoderLayer` with a final norm."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        pairwise_encoding_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_interm_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the decoder.

        Args:
            q_dim: Relation-query width.
            kv_dim: Class-embedding width.
            pairwise_encoding_dim: Input width of the pairwise spatial encoding.
            num_layers: Number of decoder layers.
            num_heads: Number of attention heads.
            ffn_interm_dim: FFN intermediate width.
            dropout: Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConsistencyDecoderLayer(
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    pairwise_encoding_dim=pairwise_encoding_dim,
                    num_heads=num_heads,
                    ffn_interm_dim=ffn_interm_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(q_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-init weight matrices (mirrors the original decoder)."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        queries: Tensor,
        features: Tensor,
        category_embedding: Tensor,
        spatial: dict[str, Tensor],
        gate: Tensor,
        high_freq_ratio: Tensor,
    ) -> Tensor:
        """Run the decoder stack.

        Args:
            queries: Relation queries ``(N, q_dim)``.
            features: Class-embedding matrix ``(num_classes, kv_dim)``.
            category_embedding: Subject class embedding ``(N, kv_dim)``.
            spatial: Spatial prior dict (see :class:`ConsistencyDecoderLayer`).
            gate: Per-query gate ``(N, 1)`` in ``[0, 1]``.
            high_freq_ratio: Per-query high-frequency ratio ``(N,)``.

        Returns:
            Final relation representations ``(N, q_dim)``.
        """
        output = queries
        for layer in self.layers:
            output = layer(
                queries=output,
                features=features,
                category_embedding=category_embedding,
                spatial=spatial,
                gate=gate,
                high_freq_ratio=high_freq_ratio,
            )
        return cast(Tensor, self.norm(output))


def build_spatial_prior(
    boxes_a: Tensor,
    boxes_b: Tensor,
    freq_diff_4d: Tensor,
    img_shape: tuple[int, int] | Tensor,
) -> dict[str, Tensor]:
    """Assemble the spatial-prior dict for the decoder.

    Args:
        boxes_a: ``(N, 4)`` pixel boxes of the clear target A.
        boxes_b: ``(N, 4)`` pixel boxes of the blurry target B.
        freq_diff_4d: ``(N, 4)`` frequency-derived PE (replaces angle dims).
        img_shape: Either an image ``(height, width)`` tuple or a ``(N, 2)``
            tensor of per-box image heights/widths.

    Returns:
        Dict with ``centre_pe`` (N, 256) and ``pairwise_feat`` (N, 32).
    """
    # Midpoint of the two boxes' *centres* (boxes are [x1, y1, x2, y2]).
    centre_a = (boxes_a[:, :2] + boxes_a[:, 2:]) / 2.0
    centre_b = (boxes_b[:, :2] + boxes_b[:, 2:]) / 2.0
    pair_centres = (centre_a + centre_b) / 2.0
    if isinstance(img_shape, tuple):
        h_t: Tensor = torch.as_tensor(img_shape[0], dtype=pair_centres.dtype, device=pair_centres.device)
        w_t: Tensor = torch.as_tensor(img_shape[1], dtype=pair_centres.dtype, device=pair_centres.device)
    else:
        h_t = img_shape[:, 0]
        w_t = img_shape[:, 1]
    centre = torch.stack([pair_centres[:, 0] / w_t, pair_centres[:, 1] / h_t], dim=-1)
    centre_pe = compute_sinusoidal_pe(centre)
    pairwise_feat = build_pairwise_pe(boxes_a, boxes_b, freq_diff_4d, img_shape)
    return {"centre_pe": centre_pe, "pairwise_feat": pairwise_feat}
