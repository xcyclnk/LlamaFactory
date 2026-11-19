"""Load timestamp metadata for a LlamaFactory video represented by JPEG frames."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence


EXPECTED_FRAME_COUNT = 128


@lru_cache(maxsize=256)
def _load_manifest(manifest_path_string: str) -> tuple[tuple[float, ...], int]:
    manifest_path = Path(manifest_path_string)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc

    timestamps_raw = manifest.get("timestamps_seconds")
    if manifest.get("num_frames") != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Expected num_frames={EXPECTED_FRAME_COUNT} in {manifest_path}, "
            f"got {manifest.get('num_frames')!r}"
        )
    if not isinstance(timestamps_raw, list) or len(timestamps_raw) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FRAME_COUNT} timestamps_seconds values in {manifest_path}"
        )

    timestamps: list[float] = []
    for index, value in enumerate(timestamps_raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Timestamp {index} in {manifest_path} is not numeric: {value!r}")
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Timestamp {index} in {manifest_path} is invalid: {value!r}")
        if timestamps and timestamp < timestamps[-1]:
            raise ValueError(f"timestamps_seconds is not nondecreasing in {manifest_path}")
        timestamps.append(timestamp)

    total_decoded_frames = manifest.get("total_decoded_frames")
    if not isinstance(total_decoded_frames, int) or total_decoded_frames < 1:
        raise ValueError(f"Invalid total_decoded_frames in {manifest_path}: {total_decoded_frames!r}")
    return tuple(timestamps), total_decoded_frames


def load_qwen3vl_video_metadata(
    sampled_frames: Sequence[str | Path],
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Return vLLM processor/video metadata using exact manifest timestamps.

    Qwen3-VL reconstructs seconds as ``frame_index / fps``.  The manifest has
    timestamps but intentionally omits source frame indices, so this function
    uses the video's average FPS to create floating-point pseudo-indices whose
    quotient recovers every recorded timestamp exactly.

    ``None`` means the frame sequence is not one of this project's pre-extracted
    videos and lets the caller retain its legacy fixed-FPS behavior.
    """
    if not sampled_frames:
        return None
    frame_paths = [Path(path) for path in sampled_frames]
    parents = {path.parent.resolve() for path in frame_paths}
    if len(parents) != 1:
        return None

    frame_dir = next(iter(parents))
    manifest_path = frame_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    if len(frame_paths) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FRAME_COUNT} pre-extracted frames in {frame_dir}, "
            f"got {len(frame_paths)}"
        )

    expected_names = [f"{position:03d}.jpg" for position in range(1, EXPECTED_FRAME_COUNT + 1)]
    actual_names = [path.name for path in frame_paths]
    if actual_names != expected_names:
        raise ValueError(
            f"Pre-extracted frames in {frame_dir} must be ordered 001.jpg through 128.jpg"
        )

    timestamps, total_decoded_frames = _load_manifest(str(manifest_path))
    duration = timestamps[-1]
    if duration > 0 and total_decoded_frames > 1:
        source_fps = (total_decoded_frames - 1) / duration
    else:
        source_fps = 1.0
    frame_indices = [timestamp * source_fps for timestamp in timestamps]

    processor_kwargs: dict[str, object] = {
        "fps": source_fps,
        "do_sample_frames": False,
        "total_num_frames": total_decoded_frames,
    }
    video_metadata: dict[str, object] = {
        "fps": source_fps,
        "frames_indices": frame_indices,
        "total_num_frames": total_decoded_frames,
        "duration": duration,
        "video_backend": "opencv",
    }
    return processor_kwargs, video_metadata

