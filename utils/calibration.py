"""
calibration.py
Calibration: raw sub-scores → IELTS official band
Dùng mapping table + np.interp thay vì hand-crafted multiplier
"""

import json
import numpy as np
from pathlib import Path

_DEFAULT_MAP = Path(__file__).parent.parent / "data" / "calibration_map.json"


def calibrate(sub_scores: dict, map_path: str = None) -> float:
    """
    Chuyển đổi raw sub-scores → IELTS overall band.

    Args:
        sub_scores: {"fluency_coherence": 6.2, "lexical_resource": 5.8, ...}
        map_path: Đường dẫn tới calibration_map.json (tuỳ chọn)

    Returns:
        IELTS overall band (làm tròn về thang 0.5)
    """
    # Lấy trung bình 4 tiêu chí
    values = list(sub_scores.values())
    raw_mean = float(np.mean(values))

    # Load calibration map
    path = Path(map_path) if map_path else _DEFAULT_MAP
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        breakpoints = data["breakpoints"]
    except Exception:
        # Fallback: không calibrate
        breakpoints = [[i * 0.5, i * 0.5] for i in range(6, 19)]

    xs = [p[0] for p in breakpoints]
    ys = [p[1] for p in breakpoints]

    # Nội suy tuyến tính
    calibrated = float(np.interp(raw_mean, xs, ys))

    # Làm tròn về thang IELTS (bước 0.5)
    official_band = round(calibrated * 2) / 2

    return official_band


def round_to_ielts_band(score: float) -> float:
    """Làm tròn điểm về thang IELTS (4.0, 4.5, 5.0, ..., 9.0)."""
    clamped = float(np.clip(score, 4.0, 9.0))
    return round(clamped * 2) / 2
