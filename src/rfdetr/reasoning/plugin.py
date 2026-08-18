# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""FFT consistency-reasoning plugin (trained, CTRP-CRDe inspired).

A trainable post-detection module that re-scores *blurry* candidate detections
``B`` using *clear* reference detections ``A`` in the same image:

- ``CandidateBuilder`` buckets the frozen detector's low-threshold candidates
  into clear ``A`` (``score >= tau_high``) and blurry ``B``
  (``conf_low <= score < conf_high``), pairs each ``B`` with its ``top_k``
  nearest ``A`` boxes, and (in training) labels each pair by matching ``B``
  against ground truth.
- ``ConsistencyReasonPlugin`` encodes each ``A``/``B`` image patch with a
  :class:`~rfdetr.reasoning.freq.FreqPatchEncoder`, derives a propagation gate
  and a frequency-derived pairwise encoding from the spectra, and runs a
  :class:`~rfdetr.reasoning.decoder.ConsistencyDecoder` that attends to the
  detector's class-embedding matrix.  The decoder's per-pair logit is turned
  into a bounded score adjustment (boost for likely-real missed objects,
  suppression for likely-false alarms) applied on top of ``B``'s raw
  confidence.

The plugin is trained with the detector frozen: candidates come from a
low-threshold pass of the baseline model, and the only learnable parameters
belong to the FFT encoder, the gate/PE MLPs and the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from rfdetr.models.math import MLP
from rfdetr.reasoning.decoder import ConsistencyDecoder, build_spatial_prior
from rfdetr.reasoning.freq import FreqPatchEncoder


@dataclass
class ReasonConfig:
    """Hyper-parameters of the FFT consistency plugin.

    Attributes:
        freq_dim: Output width of the spectral feature ``Ffreq``.
        patch_size: Fixed square size patches are resized to before the FFT.
        top_k: Number of nearest clear targets ``A`` paired with each blurry ``B``.
        tau_high: Confidence at or above which a candidate counts as a clear ``A``.
        conf_low: Lower bound of the blurry-``B`` confidence band (inclusive).
        conf_high: Upper bound of the blurry-``B`` confidence band (exclusive).
        num_heads: Attention heads in the CRDe decoder.
        num_layers: CRDe decoder layers.
        boost_scale: Maximum absolute score adjustment applied to a ``B``.
        p_threshold: The plugin's sigmoid probability above which a ``B`` is
            boosted and below which it is suppressed.  The decoder learns a
            *discriminative* probability that is not calibrated near ``1`` for
            true positives (diagnosed at ~0.2 for GT-matched B vs ~0.08 for
            false alarms), so the natural decision line is well below ``0.5``.
            ``boost = boost_scale * (p - p_threshold)``.
        mask_prob: Training-only augmentation (CTRP-style random masking).
            Fraction of clear ``A`` targets randomly masked (pixels zeroed) to
            synthesise occluded ``B`` positives, supplementing images that have
            no natural blurry candidates.  ``0`` disables.
        mask_fraction: Fraction of a masked patch's pixels set to zero.
        reason_class_ids: If set, the plugin only re-scores blurry ``B``
            candidates of these class ids at inference (other classes keep
            their baseline score untouched).  ``None`` re-scores all classes.
    """

    freq_dim: int = 64
    patch_size: int = 32
    top_k: int = 5
    tau_high: float = 0.5
    conf_low: float = 0.05
    conf_high: float = 0.3
    num_heads: int = 4
    num_layers: int = 2
    boost_scale: float = 0.5
    p_threshold: float = 0.12
    mask_prob: float = 0.1
    mask_fraction: float = 0.75
    reason_class_ids: tuple[int, ...] | None = None


class CandidateBuilder:
    """Bucket candidates into A/B and build Top-K relation pairs.

    The builder is shared by the training script (which additionally supplies ground-truth boxes for labelling) and by
    inference (which only needs the A/B split and pairing).
    """

    def __init__(self, config: ReasonConfig) -> None:
        """Initialise with the plugin config.

        Args:
            config: Plugin hyper-parameters.
        """
        self.config = config

    def split(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split candidates into clear-A and blurry-B indices.

        Args:
            boxes: Candidate boxes ``(N, 4)`` as pixel ``[x1, y1, x2, y2]``.
            scores: Candidate confidences ``(N,)``.

        Returns:
            ``(a_inds, b_inds)`` — integer arrays of A and B indices.
        """
        a_inds = np.flatnonzero(scores >= self.config.tau_high)
        b_inds = np.flatnonzero((scores >= self.config.conf_low) & (scores < self.config.conf_high))
        return a_inds, b_inds

    def pair_topk(self, boxes: np.ndarray, a_inds: np.ndarray, b_inds: np.ndarray) -> np.ndarray:
        """Pair each blurry B with its Top-K nearest clear A by centre distance.

        Args:
            boxes: Candidate boxes ``(N, 4)`` pixel ``[x1, y1, x2, y2]``.
            a_inds: Clear-A candidate indices.
            b_inds: Blurry-B candidate indices.

        Returns:
            Pair index array of shape ``(num_pairs, 2)`` where each row is
            ``(a_index, b_index)``; an empty ``(0, 2)`` array when there are no
            A's or no B's.
        """
        if a_inds.size == 0 or b_inds.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        a_c = boxes[a_inds][:, :2]
        b_c = boxes[b_inds][:, :2]
        # dist: (num_b, num_a) — L2 centre distance between every B and every A.
        dist = np.sqrt(((b_c[:, None, :] - a_c[None, :, :]) ** 2).sum(-1))
        k = min(self.config.top_k, a_inds.size)
        nearest_a = np.argsort(dist, axis=1)[:, :k]  # (num_b, k)
        pairs: list[tuple[int, int]] = []
        for bi, row in enumerate(nearest_a):
            for ai in row:
                pairs.append((int(a_inds[ai]), int(b_inds[bi])))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def label_pairs(
        self,
        b_inds: np.ndarray,
        b_boxes: np.ndarray,
        b_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> np.ndarray:
        """Label each blurry B as matched (1) or false-alarm (0) against GT.

        A B is positive when it overlaps a ground-truth box of the *same* class
        with IoU >= 0.5 (class-aware matching, matching the competition eval).

        Args:
            b_inds: Blurry-B indices.
            b_boxes: ``(num_b, 4)`` pixel boxes of the B candidates.
            b_classes: ``(num_b,)`` class ids of the B candidates.
            gt_boxes: ``(num_gt, 4)`` pixel GT boxes.
            gt_classes: ``(num_gt,)`` class ids of the GT boxes.

        Returns:
            Binary labels ``(num_b,)`` aligned with ``b_inds``.
        """
        labels = np.zeros(b_inds.size, dtype=np.float32)
        if b_inds.size == 0 or gt_boxes.size == 0:
            return labels
        for i in range(b_inds.size):
            same_class = gt_classes == b_classes[i]
            if not same_class.any():
                continue
            ious = _batch_ious(gt_boxes[same_class], b_boxes[i])
            if ious.size and ious.max() >= 0.5:
                labels[i] = 1.0
        return labels


def _batch_ious(gt_boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """IoU between one box and an array of GT boxes.

    Args:
        gt_boxes: ``(M, 4)`` pixel boxes.
        box: Single ``(4,)`` pixel box.

    Returns:
        IoU array ``(M,)``.
    """
    x1 = np.maximum(gt_boxes[:, 0], box[0])
    y1 = np.maximum(gt_boxes[:, 1], box[1])
    x2 = np.minimum(gt_boxes[:, 2], box[2])
    y2 = np.minimum(gt_boxes[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    garea = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    barea = (box[2] - box[0]) * (box[3] - box[1])
    union = garea + barea - inter
    return np.asarray(inter / np.maximum(union, 1e-6))


class ConsistencyReasonPlugin(nn.Module):
    """FFT consistency-reasoning plugin.

    Args:
        num_classes: Number of object classes (used to size the class
            embedding read, which is passed in at forward time).
        config: Plugin hyper-parameters.
    """

    def __init__(self, num_classes: int, config: ReasonConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        self.freq_encoder = FreqPatchEncoder(
            freq_dim=config.freq_dim,
            patch_size=config.patch_size,
        )
        # Gate: sigmoid over concat(Ffreq_A, Ffreq_B).  Bias initialised so the
        # gate starts near 1 (free propagation) and does not perturb the baseline.
        gate_head = nn.Linear(config.freq_dim, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.freq_dim * 2, config.freq_dim),
            nn.ReLU(),
            gate_head,
        )
        nn.init.zeros_(gate_head.bias)  # sigmoid(0) = 0.5

        # PE: |Ffreq_A - Ffreq_B| -> 4 dims replacing the angle dims.
        self.pe_mlp = MLP(config.freq_dim, config.freq_dim, 4, 2)

        # Joint-region visual encoder (RoI-feature stand-in): crops the A∪B
        # union rectangle and encodes it with a small CNN — the visual base of
        # the relation-pair feature that the original CTRP got from RoIAlign.
        self.joint_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, config.freq_dim),
            nn.ReLU(),
        )

        # Relation feature: inject all three sources —
        #   roi_joint (visual base) + spatial_encoding (geometry) + freq (new clue)
        self.rel_feat_head = nn.Sequential(
            nn.Linear(config.freq_dim + config.freq_dim * 2 + 32, config.freq_dim),
            nn.ReLU(),
        )

        self.decoder = ConsistencyDecoder(
            q_dim=config.freq_dim,
            kv_dim=256,
            pairwise_encoding_dim=32,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_interm_dim=config.freq_dim * 4,
        )
        self.classify = nn.Linear(config.freq_dim, 1)

    def forward(
        self,
        a_patches: Tensor,
        b_patches: Tensor,
        joint_patches: Tensor,
        boxes_a: Tensor,
        boxes_b: Tensor,
        class_ids_a: Tensor,
        class_embed_weight: Tensor,
        img_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-pair corrections and gates.

        The relation-pair query **injects** three sources (faithful to the
        CTRP design): the joint-region visual feature (RoI base), the pairwise
        spatial encoding (geometry), and the new frequency clue.

        Args:
            a_patches: ``(P, 3, H, W)`` image patches of the clear targets A.
            b_patches: ``(P, 3, H, W)`` image patches of the blurry targets B.
            joint_patches: ``(P, 3, H, W)`` image patches of the A∪B union
                rectangles (the RoI visual base of the relation feature).
            boxes_a: ``(P, 4)`` pixel boxes of A.
            boxes_b: ``(P, 4)`` pixel boxes of B.
            class_ids_a: ``(P,)`` class ids of A.
            class_embed_weight: ``(num_classes, hidden_dim)`` class-embedding
                matrix of the (frozen) detector, used as cross-attention KV.
            img_shape: Image ``(height, width)`` for spatial normalisation.

        Returns:
            ``(logits, gates, high_freq_ratios)``:
            - ``logits`` ``(P,)`` raw per-pair scores before sigmoid.
            - ``gates`` ``(P,)`` in ``[0, 1]`` — the propagation gates.
            - ``high_freq_ratios`` ``(P,)`` in ``[0, 1]``.
        """
        ffreq_a, _ = self.freq_encoder(a_patches)
        ffreq_b, high_freq = self.freq_encoder(b_patches)
        freq_diff_feat = torch.cat([ffreq_a, ffreq_b], dim=-1)  # (P, 2*freq_dim)

        gate_logits = self.gate_mlp(freq_diff_feat)
        gates = torch.sigmoid(gate_logits.squeeze(-1))  # (P,)

        freq_diff_4d = self.pe_mlp((ffreq_a - ffreq_b).abs())  # (P, 4)

        spatial = build_spatial_prior(boxes_a, boxes_b, freq_diff_4d, img_shape)
        # Semantic prior: subject class embedding per pair.
        if class_ids_a.numel():
            category = class_embed_weight[class_ids_a]  # (P, hidden_dim)
        else:
            category = class_embed_weight.new_zeros((0, class_embed_weight.size(1)))

        # ── inject: roi_joint (visual base) + spatial_encoding (geometry) + freq ──
        roi_joint = self.joint_encoder(joint_patches)  # (P, freq_dim)
        spatial_encoding = spatial["pairwise_feat"]  # (P, 32) geometry PE
        queries = self.rel_feat_head(torch.cat([roi_joint, spatial_encoding, freq_diff_feat], dim=-1))  # (P, freq_dim)
        # Cross-attention KV = the CURRENT image's detected clear targets A
        # (their class embeddings), so each blurry B is judged against the
        # high-confidence objects actually present in this scene — a faithful
        # "judge B by the A's detected here" design.  (The all-classes matrix
        # would let B attend to classes that never appear in the image.)
        unique_a = torch.unique(class_ids_a)
        if unique_a.numel():
            scene_features = class_embed_weight[unique_a]  # (num_unique_A, kv_dim)
        else:
            scene_features = class_embed_weight.new_zeros((0, class_embed_weight.size(1)))
        refined = self.decoder(
            queries=queries,
            features=scene_features,
            category_embedding=category,
            spatial=spatial,
            gate=gates.unsqueeze(-1),
            high_freq_ratio=high_freq,
        )
        logits = self.classify(refined).squeeze(-1)  # (P,)
        return logits, gates, high_freq

    @torch.inference_mode()
    def predict_detections(
        self,
        source_image: np.ndarray,
        candidate_boxes: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_classes: np.ndarray,
        class_names: list[str],
        class_embed_weight: Tensor,
        device: str,
        target_conf: float,
        *,
        reason_class_ids: tuple[int, ...] | None = None,
        filter_final: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Re-score blurry candidates and merge them with clear detections.

        Args:
            source_image: Original image as uint8 ``(H, W, 3)``.
            candidate_boxes: ``(N, 4)`` pixel candidate boxes.
            candidate_scores: ``(N,)`` candidate confidences.
            candidate_classes: ``(N,)`` candidate class ids.
            class_names: Class names ordered by class id (for the return).
            class_embed_weight: Frozen detector's class-embedding matrix.
            device: Torch device string for the plugin forward.
            target_conf: Final confidence threshold applied after adjustment.
            reason_class_ids: 可选的单次调用类别限制；未传入时使用
                ``config.reason_class_ids``。
            filter_final: 是否在返回前按 ``target_conf`` 筛选。批量评测会关闭
                此项，并在所有分数调整完成后统一应用逐类阈值。

        Returns:
            ``(boxes, scores, class_ids)``。当 ``filter_final=True`` 时，移除
            低于 ``target_conf`` 的候选；B 框保留其原始预测类别。
        """
        builder = CandidateBuilder(self.config)
        a_inds, b_inds = builder.split(candidate_boxes, candidate_scores)
        # Only treat candidates BELOW the final threshold as "blurry B" to
        # recover.  Candidates at/above target_conf are already output by the
        # baseline; re-scoring them could push a correct detection below the
        # threshold (mis-deleting a true positive).  Clip the B band's upper
        # bound to target_conf so the plugin only ever *adds* detections, never
        # removes ones the baseline already produced.
        if b_inds.size:
            b_inds = b_inds[candidate_scores[b_inds] < target_conf]
            # Optional class restriction: only re-score blurry B's of the listed
            # classes; everything else keeps its baseline score untouched.
            class_ids = self.config.reason_class_ids if reason_class_ids is None else reason_class_ids
            if class_ids is not None:
                b_inds = b_inds[np.isin(candidate_classes[b_inds], class_ids)]
        boxes = candidate_boxes
        scores = candidate_scores.copy()

        if b_inds.size and a_inds.size:
            pairs = builder.pair_topk(boxes, a_inds, b_inds)
            if pairs.size:
                a_idx = pairs[:, 0]
                b_idx = pairs[:, 1]  # original candidate indices of B
                # Position of each B within b_inds (which is a strict subset of
                # candidate indices), used to accumulate per-B adjustments.
                b_pos = np.searchsorted(b_inds, b_idx)
                a_patches = _crop_patches(source_image, boxes[a_idx])
                b_patches = _crop_patches(source_image, boxes[b_idx])
                joint_patches = _crop_joint_patches(source_image, boxes[a_idx], boxes[b_idx])
                a_patches_t = torch.from_numpy(a_patches).to(device)
                b_patches_t = torch.from_numpy(b_patches).to(device)
                joint_patches_t = torch.from_numpy(joint_patches).to(device)
                boxes_a_t = torch.from_numpy(boxes[a_idx].astype(np.float32)).to(device)
                boxes_b_t = torch.from_numpy(boxes[b_idx].astype(np.float32)).to(device)
                class_a_t = torch.from_numpy(candidate_classes[a_idx]).to(device)
                logits, gates, _ = self.forward(
                    a_patches_t,
                    b_patches_t,
                    joint_patches_t,
                    boxes_a_t,
                    boxes_b_t,
                    class_a_t,
                    class_embed_weight,
                    img_shape=(source_image.shape[0], source_image.shape[1]),
                )
                # Per-B adjustment: mean over its Top-K pairs.  The decoder's
                # sigmoid probability is discriminative but not calibrated near
                # 1 (GT-matched B ~0.2, false alarms ~0.08), so the natural
                # decision line is ``p_threshold`` (well below 0.5): a B with
                # p above it is boosted, below it is suppressed.
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                adjust = self.config.boost_scale * (prob - self.config.p_threshold)
                per_b = np.zeros(b_inds.size, dtype=np.float32)
                counts = np.zeros(b_inds.size, dtype=np.float32)
                np.add.at(per_b, b_pos, adjust)
                np.add.at(counts, b_pos, 1.0)
                adj_mean = np.where(counts > 0, per_b / np.maximum(counts, 1), 0.0)
                scores[b_inds] = scores[b_inds] + adj_mean

        keep = scores >= target_conf if filter_final else np.ones(scores.shape, dtype=bool)
        return boxes[keep], scores[keep], candidate_classes[keep]


def _crop_patches(source_image: np.ndarray, boxes: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """Crop and resize square patches from the source image.

    Args:
        source_image: Original image ``(H, W, 3)`` uint8.
        boxes: ``(N, 4)`` pixel boxes.
        patch_size: Square output size in pixels.

    Returns:
        Float32 patches ``(N, 3, patch_size, patch_size)`` in ``[0, 1]``.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 3, patch_size, patch_size), dtype=np.float32)
    h, w = source_image.shape[:2]
    patches: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        # Clamp to image bounds and enforce a minimum size to avoid empty crops.
        cx = max(0.0, min(float(w), (x1 + x2) / 2.0))
        cy = max(0.0, min(float(h), (y1 + y2) / 2.0))
        bw = max(2.0, float(x2 - x1))
        bh = max(2.0, float(y2 - y1))
        x1c = int(max(0.0, cx - bw / 2.0))
        y1c = int(max(0.0, cy - bh / 2.0))
        x2c = int(min(float(w), cx + bw / 2.0))
        y2c = int(min(float(h), cy + bh / 2.0))
        if x2c <= x1c or y2c <= y1c:
            x2c, y2c = x1c + 1, y1c + 1
        crop = source_image[y1c:y2c, x1c:x2c]
        # Resize to (patch_size, patch_size) via PIL (bilinear, matches F.resize).
        pil = Image.fromarray(crop).resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        patches.append(np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(patches, axis=0)


def _crop_joint_patches(
    source_image: np.ndarray,
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    patch_size: int = 64,
) -> np.ndarray:
    """Crop the A∪B union rectangles from the source image.

    The union rectangle spans both boxes (min of top-left, max of bottom-right),
    then is clamped to image bounds and resized to ``patch_size``.  This is the
    joint-region visual base the original CTRP fed to RoIAlign.

    Args:
        source_image: Original image ``(H, W, 3)`` uint8.
        boxes_a: ``(N, 4)`` pixel boxes of the clear targets A.
        boxes_b: ``(N, 4)`` pixel boxes of the blurry targets B.
        patch_size: Square output size in pixels.

    Returns:
        Float32 patches ``(N, 3, patch_size, patch_size)`` in ``[0, 1]``.
    """
    n = boxes_a.shape[0]
    if n == 0:
        return np.zeros((0, 3, patch_size, patch_size), dtype=np.float32)
    h, w = source_image.shape[:2]
    patches: list[np.ndarray] = []
    for (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) in zip(boxes_a, boxes_b):
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        x2 = max(ax2, bx2)
        y2 = max(ay2, by2)
        # Enforce a minimum size and clamp to image bounds.
        bw = max(2.0, float(x2 - x1))
        bh = max(2.0, float(y2 - y1))
        cx = max(0.0, min(float(w), (x1 + x2) / 2.0))
        cy = max(0.0, min(float(h), (y1 + y2) / 2.0))
        x1c = int(max(0.0, cx - bw / 2.0))
        y1c = int(max(0.0, cy - bh / 2.0))
        x2c = int(min(float(w), cx + bw / 2.0))
        y2c = int(min(float(h), cy + bh / 2.0))
        if x2c <= x1c or y2c <= y1c:
            x2c, y2c = x1c + 1, y1c + 1
        crop = source_image[y1c:y2c, x1c:x2c]
        pil = Image.fromarray(crop).resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        patches.append(np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(patches, axis=0)


def mask_pixels(patch: np.ndarray, mask_fraction: float = 0.75) -> np.ndarray:
    """Zero a random subset of a patch's pixels (training-only occlusion).

    Each pixel channel is independently kept with probability ``1 - mask_fraction``
    and zeroed otherwise — a Bernoulli-style mask (CTRP ``img_random_mask``
    used ``masked_fill(value=0)`` over a fraction of GT boxes; here we adapt it
    to a single patch).  The patch shape/dtype are preserved.

    Args:
        patch: ``(H, W, 3)`` uint8 or float image patch.
        mask_fraction: Fraction of pixels to zero (``0.75`` = keep 25%).

    Returns:
        A copy of *patch* with a random subset of pixels set to zero.

    Examples:
        >>> import numpy as np
        >>> p = np.full((8, 8, 3), 255, dtype=np.uint8)
        >>> m = mask_pixels(p, mask_fraction=0.75)
        >>> m.shape == p.shape and m.dtype == p.dtype
        True
        >>> z = float((m == 0).mean()); 0.6 < z < 0.9  # ~75% zeroed
        True
        >>> float((mask_pixels(p, 0.0) == 0).mean())  # mask_fraction=0 keeps all
        0.0
    """
    # Per-pixel-channel Bernoulli mask: each element zeroed w.p. mask_fraction.
    mask = np.random.random(patch.shape) < mask_fraction
    masked = patch.copy()
    masked[mask] = 0
    return np.asarray(masked)


class PluginLoader:
    """Load/save a :class:`ConsistencyReasonPlugin` checkpoint.

    Checkpoints are plain ``torch.save`` dicts ``{"state_dict", "config", "num_classes"}``, kept deliberately small and
    independent of the detector checkpoint format.
    """

    @staticmethod
    def save(
        plugin: ConsistencyReasonPlugin,
        path: str | Path,
    ) -> None:
        """Persist the plugin.

        Args:
            plugin: The plugin to save.
            path: Destination ``*.pth`` path.
        """
        torch.save(
            {
                "state_dict": plugin.state_dict(),
                "config": plugin.config.__dict__,
                "num_classes": plugin.num_classes,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path, num_classes: int | None = None) -> ConsistencyReasonPlugin:
        """Load a plugin.

        Args:
            path: Source ``*.pth`` path.
            num_classes: Optional class-count override; when ``None`` the saved
                count is used.

        Returns:
            The restored plugin (evaluation mode).

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the checkpoint is malformed.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ReasonConfig(**ckpt["config"])
        num = ckpt.get("num_classes") if num_classes is None else num_classes
        if num is None:
            raise ValueError("checkpoint has no num_classes and none was provided")
        plugin = ConsistencyReasonPlugin(num, config)
        plugin.load_state_dict(ckpt["state_dict"])
        plugin.eval()
        return plugin
