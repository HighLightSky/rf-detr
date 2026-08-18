# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for the FFT consistency-reasoning plugin (rfdetr.reasoning)."""

import numpy as np
import pytest
import torch

from rfdetr.reasoning import (
    CandidateBuilder,
    ConsistencyDecoder,
    ConsistencyReasonPlugin,
    FreqPatchEncoder,
    PluginLoader,
    ReasonConfig,
    build_pairwise_pe,
    compute_sinusoidal_pe,
)
from rfdetr.reasoning.plugin import mask_pixels

# ---------------------------------------------------------------------------
# FreqPatchEncoder
# ---------------------------------------------------------------------------


class TestFreqPatchEncoder:
    """Tests for the FFT patch encoder and its frequency mask."""

    def test_variable_size_patches_to_fixed_freq(self) -> None:
        """Patches of unequal size map to a fixed-dim Ffreq regardless of input H/W."""
        enc = FreqPatchEncoder(freq_dim=64, patch_size=32)
        patches = torch.rand(4, 3, 20, 30)  # non-square, non-matching sizes
        ffreq, high_freq = enc(patches)
        assert ffreq.shape == (4, 64)
        assert high_freq.shape == (4,)

    def test_freq_mask_identity_init(self) -> None:
        """The learnable frequency filter starts as all-ones (identity)."""
        enc = FreqPatchEncoder(patch_size=32)
        assert torch.allclose(enc.freq_mask.data, torch.ones_like(enc.freq_mask.data))

    def test_high_freq_ratio_in_unit_range(self) -> None:
        """The high-frequency energy ratio is bounded to [0, 1]."""
        enc = FreqPatchEncoder(patch_size=32)
        patches = torch.rand(8, 3, 40, 40)
        _, ratio = enc(patches)
        assert bool((ratio >= 0.0).all() and (ratio <= 1.0).all())

    def test_runs_in_float16_by_casting_to_fp32(self) -> None:
        """Rfft2 rejects half; the encoder accepts fp16 input and runs the FFT in fp32."""
        enc = FreqPatchEncoder(patch_size=32)
        patches = torch.rand(2, 3, 24, 24, dtype=torch.float16)
        ffreq, _ = enc(patches)
        assert ffreq.dtype == torch.float32  # spectral pipeline stays fp32

    def test_constant_patch_gives_low_high_freq_ratio(self) -> None:
        """A flat (constant) patch has almost no high-frequency energy."""
        enc = FreqPatchEncoder(patch_size=32)
        flat = torch.full((1, 3, 32, 32), 0.5)
        _, ratio = enc(flat)
        assert float(ratio[0]) < 0.1

    def test_empty_batch_does_not_crash(self) -> None:
        """torch.fft.rfft2 rejects an empty batch; the encoder must short-circuit."""
        enc = FreqPatchEncoder(patch_size=32)
        empty = torch.zeros(0, 3, 32, 32)
        ffreq, ratio = enc(empty)
        assert ffreq.shape == (0, enc.freq_dim)
        assert ratio.shape == (0,)


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------


class TestPairwisePe:
    """Tests for the formula-10 port with frequency-replaced angle dims."""

    def test_output_is_32_dim(self) -> None:
        """The pairwise encoding is 32-dim like the original bbox_with_relative mode."""
        boxes_a = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        boxes_b = torch.tensor([[150.0, 150.0, 250.0, 250.0]])
        freq_diff = torch.rand(1, 4)
        pe = build_pairwise_pe(boxes_a, boxes_b, freq_diff, img_shape=(791, 886))
        assert pe.shape == (1, 32)

    def test_angle_dims_replaced_by_frequency(self) -> None:
        """Indices 8:12 of the raw 16 carry the frequency vector, not sin/cos of theta."""
        boxes_a = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        boxes_b = torch.tensor([[150.0, 150.0, 250.0, 250.0]])
        freq_diff = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        pe = build_pairwise_pe(boxes_a, boxes_b, freq_diff, img_shape=(791, 886))
        # pe = [raw16, log(raw16)]; raw16's dims 8:12 == freq_diff.
        raw = pe[0, :16]
        assert torch.allclose(raw[8:12], torch.tensor([0.1, 0.2, 0.3, 0.4]), atol=1e-6)

    def test_different_freq_vectors_give_different_encodings(self) -> None:
        """Changing the frequency diff changes the output (freq dims are used)."""
        boxes_a = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        boxes_b = torch.tensor([[150.0, 150.0, 250.0, 250.0]])
        pe1 = build_pairwise_pe(boxes_a, boxes_b, torch.zeros(1, 4), img_shape=(791, 886))
        pe2 = build_pairwise_pe(boxes_a, boxes_b, torch.ones(1, 4), img_shape=(791, 886))
        assert not torch.allclose(pe1, pe2)

    def test_sinusoidal_pe_shape(self) -> None:
        """compute_sinusoidal_pe returns 256 dims for 2D normalised points."""
        pos = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
        pe = compute_sinusoidal_pe(pos)
        assert pe.shape == (2, 256)

    def test_per_sample_img_shape(self) -> None:
        """A (N, 2) img_shape tensor normalises each pair with its own image size."""
        boxes_a = torch.tensor([[100.0, 100.0, 200.0, 200.0], [50.0, 50.0, 150.0, 150.0]])
        boxes_b = torch.tensor([[150.0, 150.0, 250.0, 250.0], [60.0, 60.0, 160.0, 160.0]])
        freq_diff = torch.rand(2, 4)
        img_shape = torch.tensor([[791.0, 886.0], [400.0, 500.0]])
        pe = build_pairwise_pe(boxes_a, boxes_b, freq_diff, img_shape=img_shape)
        assert pe.shape == (2, 32)

    def test_degenerate_boxes_produce_no_nan(self) -> None:
        """Negative-width (unvalidated) boxes must not produce NaN in the log copy."""
        # x1 > x2 and y1 > y2 => negative width/height.
        boxes_a = torch.tensor([[99.0, 153.0, 17.0, 26.0]])
        boxes_b = torch.tensor([[35.0, 53.0, 30.0, 6.0]])
        freq_diff = torch.rand(1, 4)
        pe = build_pairwise_pe(boxes_a, boxes_b, freq_diff, img_shape=(500, 600))
        assert bool(torch.isfinite(pe).all())


# ---------------------------------------------------------------------------
# CRDe decoder
# ---------------------------------------------------------------------------


class TestConsistencyDecoder:
    """Tests for the CRDe decoder and its gate injection."""

    def test_forward_shape(self) -> None:
        """The decoder maps (N, q_dim) queries to (N, q_dim) outputs."""
        dec = ConsistencyDecoder(
            q_dim=64, kv_dim=256, pairwise_encoding_dim=32, num_layers=2, num_heads=4, ffn_interm_dim=256
        )
        n = 5
        queries = torch.randn(n, 64)
        features = torch.randn(25, 256)  # class embeddings
        category = torch.randn(n, 256)
        centre_pe = compute_sinusoidal_pe(torch.rand(n, 2))
        pairwise = torch.rand(n, 32)
        spatial = {"centre_pe": centre_pe, "pairwise_feat": pairwise}
        gate = torch.full((n, 1), 0.5)
        high_freq = torch.rand(n)
        out = dec(queries, features, category, spatial, gate, high_freq)
        assert out.shape == (n, 64)

    def test_zero_gate_suppresses_output(self) -> None:
        """A gate of 0 kills the cross-attention propagation (output small)."""
        dec = ConsistencyDecoder(
            q_dim=64, kv_dim=256, pairwise_encoding_dim=32, num_layers=1, num_heads=4, ffn_interm_dim=256
        )
        n = 4
        queries = torch.randn(n, 64)
        features = torch.randn(25, 256)
        category = torch.randn(n, 256)
        centre_pe = compute_sinusoidal_pe(torch.rand(n, 2))
        pairwise = torch.rand(n, 32)
        spatial = {"centre_pe": centre_pe, "pairwise_feat": pairwise}
        hf = torch.rand(n)

        out_zero = dec(queries.clone(), features, category, spatial, torch.zeros(n, 1), hf)
        out_one = dec(queries.clone(), features, category, spatial, torch.ones(n, 1), hf)
        # Gate 0 => cross-attention weights are zeroed => output shrinks toward
        # the residual (queries) rather than propagating class-embedding content.
        dev_zero = float((out_zero.detach() - queries).abs().mean())
        dev_one = float((out_one.detach() - queries).abs().mean())
        assert dev_zero < dev_one


# ---------------------------------------------------------------------------
# CandidateBuilder
# ---------------------------------------------------------------------------


class TestCandidateBuilder:
    """Tests for the A/B bucketing, Top-K pairing and GT labelling."""

    @pytest.fixture()
    def builder(self) -> CandidateBuilder:
        return CandidateBuilder(ReasonConfig())

    def test_split_buckets_by_confidence(self, builder: CandidateBuilder) -> None:
        """High scores are clear A; the middle band is blurry B."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32)
        scores = np.array([0.7, 0.1, 0.2])
        a_inds, b_inds = builder.split(boxes, scores)
        assert list(a_inds) == [0]
        assert list(b_inds) == [1, 2]

    def test_split_excludes_below_conf_low(self, builder: CandidateBuilder) -> None:
        """Candidates below conf_low are not B (treated as pure noise)."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32)
        scores = np.array([0.7, 0.04, 0.2])
        _, b_inds = builder.split(boxes, scores)
        assert list(b_inds) == [2]

    def test_pair_topk_uses_nearest_centres(self, builder: CandidateBuilder) -> None:
        """Each B pairs with the nearest A by centre distance."""
        # A at (5,5); Bs at (25,25) and (6,5).
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [2, 2, 10, 8]], dtype=np.float32)
        scores = np.array([0.7, 0.1, 0.2])
        a_inds, b_inds = builder.split(boxes, scores)
        pairs = builder.pair_topk(boxes, a_inds, b_inds)
        # B=2 ((2,2)-(10,8), centre ~(6,5)) is closer to A=0 than B=1.
        b2_rows = pairs[pairs[:, 1] == 2]
        b1_rows = pairs[pairs[:, 1] == 1]
        assert b2_rows[0, 0] == 0 and b1_rows[0, 0] == 0

    def test_pair_topk_empty_without_a(self, builder: CandidateBuilder) -> None:
        """No A's -> no pairs."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.1, 0.2])
        a_inds, b_inds = builder.split(boxes, scores)
        pairs = builder.pair_topk(boxes, a_inds, b_inds)
        assert pairs.shape == (0, 2)

    def test_label_pairs_class_aware_iou(self, builder: CandidateBuilder) -> None:
        """B matched to a same-class GT box with IoU>=0.5 is labelled 1, else 0."""
        b_inds = np.array([1, 2])
        b_boxes = np.array([[20, 20, 30, 30], [60, 60, 70, 70]], dtype=np.float32)
        b_classes = np.array([5, 5])
        gt_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
        gt_classes = np.array([5])
        labels = builder.label_pairs(b_inds, b_boxes, b_classes, gt_boxes, gt_classes)
        assert list(labels) == [1.0, 0.0]

    def test_label_pairs_ignores_wrong_class(self, builder: CandidateBuilder) -> None:
        """A GT box of a different class does not label B positive."""
        b_inds = np.array([1])
        b_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
        b_classes = np.array([5])
        gt_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
        gt_classes = np.array([3])
        labels = builder.label_pairs(b_inds, b_boxes, b_classes, gt_boxes, gt_classes)
        assert list(labels) == [0.0]


# ---------------------------------------------------------------------------
# ConsistencyReasonPlugin end-to-end
# ---------------------------------------------------------------------------


class TestConsistencyReasonPlugin:
    """Tests for the assembled plugin and its checkpoint round-trip."""

    @pytest.fixture()
    def plugin(self) -> ConsistencyReasonPlugin:
        return ConsistencyReasonPlugin(num_classes=25, config=ReasonConfig())

    def test_forward_returns_logits_gates_ratio(self, plugin: ConsistencyReasonPlugin) -> None:
        """The plugin emits per-pair logits, gates in [0,1] and high-freq ratios."""
        n = 3
        a_patches = torch.rand(n, 3, 40, 40)
        b_patches = torch.rand(n, 3, 40, 40)
        joint_patches = torch.rand(n, 3, 40, 40)
        boxes_a = torch.rand(n, 4) * 500 + 100
        boxes_b = torch.rand(n, 4) * 500 + 100
        class_a = torch.randint(0, 25, (n,))
        class_embed_weight = torch.randn(26, 256)  # 25 + background slot
        logits, gates, high_freq = plugin(
            a_patches,
            b_patches,
            joint_patches,
            boxes_a,
            boxes_b,
            class_a,
            class_embed_weight,
            img_shape=(791, 886),
        )
        assert logits.shape == (n,)
        assert gates.shape == (n,)
        assert high_freq.shape == (n,)
        assert bool((gates >= 0.0).all() and (gates <= 1.0).all())

    def test_empty_forward_does_not_crash(self, plugin: ConsistencyReasonPlugin) -> None:
        """An empty pair batch must not hit the rfft2 MKL error or crash downstream."""
        n = 0
        a = torch.zeros(n, 3, 32, 32)
        b = torch.zeros(n, 3, 32, 32)
        j = torch.zeros(n, 3, 32, 32)
        ba = torch.zeros(n, 4)
        bb = torch.zeros(n, 4)
        ca = torch.zeros(n, dtype=torch.long)
        ce = torch.randn(26, 256)
        logits, gates, high_freq = plugin(a, b, j, ba, bb, ca, ce, img_shape=(100, 100))
        assert logits.shape == (0,)
        assert gates.shape == (0,)
        assert high_freq.shape == (0,)

    def test_forward_outputs_are_finite(self, plugin: ConsistencyReasonPlugin) -> None:
        """Degenerate box geometry must not produce NaN/Inf in the plugin outputs."""
        torch.manual_seed(0)
        n = 8
        a = torch.rand(n, 3, 32, 32)
        b = torch.rand(n, 3, 32, 32)
        j = torch.rand(n, 3, 32, 32)
        ba = torch.rand(n, 4) * 200  # unvalidated, may be degenerate
        bb = torch.rand(n, 4) * 200
        ca = torch.randint(0, 25, (n,))
        ce = torch.randn(26, 256)
        logits, gates, high_freq = plugin(a, b, j, ba, bb, ca, ce, img_shape=(500, 600))
        assert bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(gates).all())
        assert bool(torch.isfinite(high_freq).all())

    def test_checkpoint_roundtrip(self, plugin: ConsistencyReasonPlugin, tmp_path) -> None:
        """Plugin save/load restores config and weights."""
        path = tmp_path / "plugin.pth"
        PluginLoader.save(plugin, path)
        loaded = PluginLoader.load(path)
        assert loaded.num_classes == 25
        assert loaded.config.freq_dim == plugin.config.freq_dim
        # Weights match.
        for (n1, p1), (n2, p2) in zip(plugin.named_parameters(), loaded.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_predict_detections_re_scores(self, plugin: ConsistencyReasonPlugin) -> None:
        """predict_detections returns filtered boxes/scores/classes and boosts B."""
        image = (np.random.rand(600, 700, 3) * 255).astype(np.uint8)
        boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300], [50, 50, 150, 150]], dtype=np.float32)
        scores = np.array([0.7, 0.1, 0.2], dtype=np.float32)
        classes = np.array([0, 5, 5], dtype=np.int64)
        out_boxes, out_scores, out_classes = plugin.predict_detections(
            source_image=image,
            candidate_boxes=boxes,
            candidate_scores=scores,
            candidate_classes=classes,
            class_names=[],
            class_embed_weight=torch.randn(26, 256),
            device="cpu",
            target_conf=0.25,
        )
        assert out_boxes.shape[1] == 4
        assert out_scores.shape == (len(out_boxes),)
        assert out_classes.shape == (len(out_boxes),)
        # The clear A detection survives; scores are at/above the threshold.
        assert bool((out_scores >= 0.25).all())


# ---------------------------------------------------------------------------
# mask_pixels (training-only synthetic-B occlusion)
# ---------------------------------------------------------------------------


class TestMaskPixels:
    """Tests for the CTRP-style random-masking helper."""

    def test_shape_and_dtype_preserved(self) -> None:
        """Masking preserves the patch shape and dtype."""
        patch = np.full((16, 16, 3), 255, dtype=np.uint8)
        masked = mask_pixels(patch, mask_fraction=0.75)
        assert masked.shape == patch.shape
        assert masked.dtype == patch.dtype

    def test_zeroes_roughly_mask_fraction(self) -> None:
        """With a large patch the zeroed fraction is close to mask_fraction."""
        # Use a large patch so the Bernoulli fraction is precise.
        patch = np.ones((64, 64, 3), dtype=np.float32) * 255.0
        frac = 0.4
        # Deterministic via a large sample size (fraction converges).
        masked = mask_pixels(patch, mask_fraction=frac)
        zero_frac = float((masked == 0).mean())
        assert 0.3 < zero_frac < 0.5

    def test_zero_fraction_keeps_all(self) -> None:
        """mask_fraction=0 leaves every pixel intact."""
        patch = np.full((8, 8, 3), 255, dtype=np.uint8)
        masked = mask_pixels(patch, mask_fraction=0.0)
        assert float((masked == 0).mean()) == 0.0

    def test_one_fraction_zeroes_all(self) -> None:
        """mask_fraction=1 zeros every pixel."""
        patch = np.full((8, 8, 3), 255, dtype=np.uint8)
        masked = mask_pixels(patch, mask_fraction=1.0)
        assert float((masked == 0).mean()) == 1.0
