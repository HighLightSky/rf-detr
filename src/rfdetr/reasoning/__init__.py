# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""FFT consistency-reasoning plugin (CTRP-CRDe inspired, trained).

A trainable post-detection module that re-scores *blurry* candidate
detections ``B`` using *clear* reference detections ``A`` in the same image.
The module is a light port of the CTRP "consistency reasoning" decoder
(Sun et al., IEEE TIP) adapted to RF-DETR's horizontal-box transformer
detector:

- :mod:`~rfdetr.reasoning.freq` — FFT patch encoder with a learnable
  frequency-domain filter (produces ``Ffreq``).
- :mod:`~rfdetr.reasoning.pe` — pairwise spatial position encoding where the
  original angle dims (meaningless for horizontal boxes) are replaced by a
  frequency-derived 4-vector.
- :mod:`~rfdetr.reasoning.decoder` — CRDe decoder with a dynamic propagation
  gate injected into the cross-attention weights.
- :mod:`~rfdetr.reasoning.plugin` — ``ConsistencyReasonPlugin`` orchestrating
  candidate bucketing, Top-K pairing, encoding and re-scoring.
"""

from rfdetr.reasoning.decoder import ConsistencyDecoder, ConsistencyDecoderLayer
from rfdetr.reasoning.freq import FreqPatchEncoder
from rfdetr.reasoning.pe import build_pairwise_pe, compute_sinusoidal_pe
from rfdetr.reasoning.plugin import CandidateBuilder, ConsistencyReasonPlugin, PluginLoader, ReasonConfig

__all__ = [
    "CandidateBuilder",
    "ConsistencyDecoder",
    "ConsistencyDecoderLayer",
    "ConsistencyReasonPlugin",
    "FreqPatchEncoder",
    "PluginLoader",
    "ReasonConfig",
    "build_pairwise_pe",
    "compute_sinusoidal_pe",
]
