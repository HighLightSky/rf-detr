# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Raw dataset sample cache helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from rfdetr.utilities.logger import get_logger

logger = get_logger()
RAW_IMAGE_CACHE_VERSION = 1


def fingerprint_path(path: str | Path) -> str:
    """Return a stable fingerprint for a path's current file identity.

    Args:
        path: File path to fingerprint.

    Returns:
        Short SHA256 digest including absolute path, size, and mtime when available.
    """
    resolved = os.path.realpath(os.fspath(path))
    hasher = hashlib.sha256()
    hasher.update(resolved.encode("utf-8", errors="surrogateescape"))
    try:
        stat = os.stat(resolved)
    except OSError:
        hasher.update(b":missing")
    else:
        hasher.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()[:24]


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array.

    Args:
        path: Image file path.

    Returns:
        RGB image array with shape ``[H, W, 3]``.
    """
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image from path: {os.fspath(path)}") from exc


class RawImageCache:
    """Disk cache for decoded RGB images.

    The cache stores only deterministic raw image decoding results. Random training transforms, Mosaic layout, resize,
    normalization, and target conversion continue to run through the existing dataset path.
    """

    def __init__(self, cache_dir: str | Path, *, rebuild: bool = False) -> None:
        """Create a raw-image cache.

        Args:
            cache_dir: Directory that stores per-image ``.npz`` files.
            rebuild: When ``True``, remove stale ``.npz`` files in this cache directory before use.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if rebuild:
            for cache_file in self.cache_dir.glob("*.npz"):
                try:
                    cache_file.unlink()
                except OSError:
                    logger.warning("Could not remove stale dataset cache file %s", cache_file)

    def load_or_create(
        self,
        image_path: str | Path,
        loader: Callable[[str | Path], np.ndarray] = load_rgb_image,
    ) -> np.ndarray:
        """Load a decoded image from cache or create it from the source image.

        Args:
            image_path: Source image path.
            loader: Callable used on cache miss.

        Returns:
            RGB image array. The returned array is writable and detached from the cache file.
        """
        cache_path = self.cache_dir / f"{fingerprint_path(image_path)}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    return np.array(data["image"], copy=True)
            except (OSError, ValueError, KeyError):
                logger.warning("Ignoring corrupt dataset cache file %s", cache_path)

        image = loader(image_path)
        image = np.asarray(image, dtype=np.uint8)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                np.savez(tmp_file, version=np.array([RAW_IMAGE_CACHE_VERSION], dtype=np.int64), image=image)
            os.replace(tmp_path, cache_path)
        except OSError:
            logger.warning("Could not write dataset cache file %s", cache_path)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return np.array(image, copy=True)
