# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图 proxy 和 ROI 读取后端。"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_FALLBACK_WARNED = False
_PYVIPS_AVAILABLE: bool | None = None
_PYVIPS_IMPORT_ERROR: Exception | None = None


def _check_pyvips() -> bool:
    """只探测一次 pyvips，避免每个大图重复触发 ctypes 动态库查找。"""
    global _PYVIPS_AVAILABLE, _PYVIPS_IMPORT_ERROR
    if _PYVIPS_AVAILABLE is not None:
        return _PYVIPS_AVAILABLE
    try:
        import pyvips  # noqa: F401
    except Exception as exc:
        _PYVIPS_AVAILABLE = False
        _PYVIPS_IMPORT_ERROR = exc
    else:
        _PYVIPS_AVAILABLE = True
    return _PYVIPS_AVAILABLE


def _cache_path(cache_dir: Path | None, image_path: Path, max_side: int) -> Path | None:
    """生成带路径哈希和尺寸参数的 proxy 缓存路径。"""
    if cache_dir is None:
        return None
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}_{max_side}.npz"


class OpenCVImageSource:
    """使用 OpenCV 完整解码的兼容图像源。"""

    backend_name = "opencv"
    used_fallback = False

    def __init__(self, image_path: str | Path, cache_dir: str | Path | None = None) -> None:
        del cache_dir
        self.image_path = Path(image_path)
        self._rgb: np.ndarray | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> np.ndarray:
        if self._rgb is None:
            with self._load_lock:
                if self._rgb is None:
                    image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        raise FileNotFoundError(f"无法读取图像: {self.image_path}")
                    self._rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._rgb

    @property
    def size(self) -> tuple[int, int]:
        image = self._load()
        return image.shape[1], image.shape[0]

    def read_proxy(self, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
        image = self._load()
        height, width = image.shape[:2]
        scale = min(1.0, float(max_side) / max(height, width))
        if scale == 1.0:
            return np.ascontiguousarray(image), (width, height)
        resized = cv2.resize(
            image,
            (max(round(width * scale), 1), max(round(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized), (width, height)

    def read_roi(
        self,
        xyxy: tuple[int, int, int, int],
        output_size: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        image = self._load()
        height, width = image.shape[:2]
        x0, y0, x1, y1 = xyxy
        x0 = max(min(int(x0), width - 1), 0)
        y0 = max(min(int(y0), height - 1), 0)
        x1 = max(min(int(x1), width), x0 + 1)
        y1 = max(min(int(y1), height), y0 + 1)
        roi = np.ascontiguousarray(image[y0:y1, x0:x1])
        roi_size = (x1 - x0, y1 - y0)
        if output_size is not None:
            roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return roi, roi_size


class PyvipsImageSource:
    """使用 libvips 惰性读取 proxy 和 ROI 的图像源。"""

    backend_name = "pyvips"
    used_fallback = False

    def __init__(self, image_path: str | Path, cache_dir: str | Path | None = None) -> None:
        self.image_path = Path(image_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._image: Any | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> Any:
        if self._image is None:
            with self._load_lock:
                if self._image is None:
                    import pyvips

                    image = pyvips.Image.new_from_file(str(self.image_path), access="sequential")
                    if image.bands > 3:
                        image = image.extract_band(0, n=3)
                    elif image.bands == 1:
                        image = image.bandjoin([image, image])
                    self._image = image
        return self._image

    @staticmethod
    def _to_numpy(image: Any) -> np.ndarray:
        array = np.frombuffer(image.write_to_memory(), dtype=np.uint8)
        return np.ascontiguousarray(array.reshape(image.height, image.width, image.bands))

    @property
    def size(self) -> tuple[int, int]:
        image = self._load()
        return int(image.width), int(image.height)

    def read_proxy(self, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
        original_size = self.size
        cached = _cache_path(self.cache_dir, self.image_path, max_side)
        if cached is not None and cached.exists():
            with np.load(cached) as data:
                return np.ascontiguousarray(data["proxy"]), original_size
        image = self._load()
        scale = min(1.0, float(max_side) / max(image.width, image.height))
        proxy = image.resize(scale) if scale < 1.0 else image
        array = self._to_numpy(proxy)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, proxy=array)
        return array, original_size

    def read_roi(
        self,
        xyxy: tuple[int, int, int, int],
        output_size: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        image = self._load()
        x0, y0, x1, y1 = xyxy
        x0 = max(min(int(x0), image.width - 1), 0)
        y0 = max(min(int(y0), image.height - 1), 0)
        x1 = max(min(int(x1), image.width), x0 + 1)
        y1 = max(min(int(y1), image.height), y0 + 1)
        roi = image.crop(x0, y0, x1 - x0, y1 - y0)
        roi_size = (x1 - x0, y1 - y0)
        if output_size is not None:
            roi = roi.resize(float(output_size) / roi.width, vscale=float(output_size) / roi.height)
        return self._to_numpy(roi), roi_size


def create_image_source(
    image_path: str | Path,
    backend: str = "auto",
    cache_dir: str | Path | None = None,
    strict: bool = False,
) -> OpenCVImageSource | PyvipsImageSource:
    """按配置创建图像源，默认优先使用 pyvips。"""
    normalized = backend.lower()
    if normalized not in {"auto", "pyvips", "opencv"}:
        raise ValueError(f"roi_backend 必须为 auto、pyvips 或 opencv，实际为 {backend!r}")
    if normalized == "opencv":
        return OpenCVImageSource(image_path, cache_dir)
    global _FALLBACK_WARNED
    if not _check_pyvips():
        exc = _PYVIPS_IMPORT_ERROR
        if normalized == "pyvips" and strict:
            raise RuntimeError("roi_backend=pyvips 但当前环境不可用 libvips/pyvips") from exc
        if not _FALLBACK_WARNED:
            print(f"[w] pyvips 不可用，图像回退 OpenCV: {exc}", flush=True)
            _FALLBACK_WARNED = True
        source = OpenCVImageSource(image_path, cache_dir)
        source.used_fallback = True
        return source
    return PyvipsImageSource(image_path, cache_dir)
