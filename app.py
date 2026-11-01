from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

try:
    import cv2
except Exception as exc:  # pragma: no cover
    st.set_page_config(page_title="Skin Lesion Classification Assistant")
    st.error(
        "OpenCV is required to generate the demo video. "
        "Install dependencies from requirements.txt and restart Streamlit."
    )
    raise RuntimeError("OpenCV import failed") from exc

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


RiskCategory = Literal["Low", "Moderate", "High"]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


# -------------------------------------------------------------------------
# Configuration. Keep operational thresholds and paths environment-driven.
# -------------------------------------------------------------------------
DEFAULT_LOW_THRESHOLD = _env_float("SKIN_RISK_LOW_THRESHOLD", 0.25)
DEFAULT_HIGH_THRESHOLD = _env_float("SKIN_RISK_HIGH_THRESHOLD", 0.70)

DEFAULT_FPS = _env_int("SKIN_DEMO_FPS", 12)
DEFAULT_FRAME_COUNT = _env_int("SKIN_DEMO_FRAMES", 90)
DEFAULT_WIDTH = _env_int("SKIN_DEMO_WIDTH", 960)
DEFAULT_HEIGHT = _env_int("SKIN_DEMO_HEIGHT", 540)

OUTPUT_DIR = Path(os.getenv("SKIN_DEMO_OUTPUT_DIR", tempfile.gettempdir()))
SCHEMA_VERSION = os.getenv("SKIN_OUTPUT_SCHEMA_VERSION", "0.1.0-demo")
MODEL_VERSION = os.getenv("SKIN_MODEL_VERSION", "synthetic-heuristic-v0")

# Demo heuristic coefficients. In a production system these would come from
# a validated model artifact and configuration store, not from application code.
MODEL_INTERCEPT = _env_float("SKIN_MODEL_INTERCEPT", -3.10)
MODEL_WEIGHT_ASYM = _env_float("SKIN_MODEL_WEIGHT_ASYM", 1.90)
MODEL_WEIGHT_BORDER = _env_float("SKIN_MODEL_WEIGHT_BORDER", 1.70)
MODEL_WEIGHT_COLOR = _env_float("SKIN_MODEL_WEIGHT_COLOR", 2.10)
MODEL_WEIGHT_DIAMETER = _env_float("SKIN_MODEL_WEIGHT_DIAMETER", 1.40)
MODEL_WEIGHT_EVOLUTION = _env_float("SKIN_MODEL_WEIGHT_EVOLUTION", 2.60)
MODEL_WEIGHT_ULCERATION = _env_float("SKIN_MODEL_WEIGHT_ULCERATION", 1.50)
MODEL_WEIGHT_ATYPICAL_NETWORK = _env_float("SKIN_MODEL_WEIGHT_ATYPICAL_NETWORK", 1.80)

DIAMETER_CENTER_MM = _env_float("SKIN_DIAMETER_CENTER_MM", 4.0)
DIAMETER_SCALE_MM = _env_float("SKIN_DIAMETER_SCALE_MM", 6.0)

FEATURE_ALERT_THRESHOLD = _env_float("SKIN_FEATURE_ALERT_THRESHOLD", 0.60)
DIAMETER_ALERT_MM = _env_float("SKIN_DIAMETER_ALERT_MM", 6.0)
EVOLUTION_ALERT_THRESHOLD = _env_float("SKIN_EVOLUTION_ALERT_THRESHOLD", 0.50)

VIDEO_WRITER_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("avc1", ".mp4", "video/mp4"),
    ("H264", ".mp4", "video/mp4"),
    ("VP90", ".webm", "video/webm"),
    ("VP80", ".webm", "video/webm"),
    ("mp4v", ".mp4", "video/mp4"),
)


# -------------------------------------------------------------------------
# Domain objects
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class LesionFeatures:
    asymmetry: float
    border_irregularity: float
    color_variegation: float
    diameter_mm: float
    evolution_score: float
    ulceration: bool
    atypical_network: bool


@dataclass(frozen=True)
class Prediction:
    risk_probability: float
    risk_category: RiskCategory
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    label: str
    suspected_condition: str
    skin_tone: int
    features: LesionFeatures
    center_xy: tuple[float, float]
    axes_frac: tuple[float, float]
    angle_deg: float
    palette: tuple[tuple[int, int, int], ...]


SKIN_TONE_COLORS: dict[int, tuple[int, int, int]] = {
    1: (246, 226, 208),
    2: (232, 205, 181),
    3: (205, 170, 138),
    4: (177, 134, 96),
    5: (136, 91, 58),
    6: (94, 60, 39),
}


# -------------------------------------------------------------------------
# Deterministic helpers
# -------------------------------------------------------------------------
def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _make_demo_cases() -> tuple[DemoCase, ...]:
    """
    Synthetic, visually plausible demo cases.

    These are not patient data. They are intended only to exercise the UI,
    video generation, output contract, and human-factors messaging.
    """
    return (
        DemoCase(
            case_id="SYN-001",
            label="Benign acquired nevus",
            suspected_condition="Benign melanocytic nevus",
            skin_tone=3,
            features=LesionFeatures(
                asymmetry=0.18,
                border_irregularity=0.22,
                color_variegation=0.25,
                diameter_mm=4.2,
                evolution_score=0.08,
                ulceration=False,
                atypical_network=False,
            ),
            center_xy=(0.48, 0.52),
            axes_frac=(0.13, 0.10),
            angle_deg=12.0,
            palette=((95, 58, 36), (120, 76, 45), (74, 45, 28)),
        ),
        DemoCase(
            case_id="SYN-002",
            label="Seborrheic keratosis",
            suspected_condition="Seborrheic keratosis",
            skin_tone=4,
            features=LesionFeatures(
                asymmetry=0.35,
                border_irregularity=0.45,
                color_variegation=0.45,
                diameter_mm=6.0,
                evolution_score=0.05,
                ulceration=False,
                atypical_network=False,
            ),
            center_xy=(0.55, 0.45),
            axes_frac=(0.16, 0.12),
            angle_deg=-18.0,
            palette=(
                (112, 78, 52),
                (70, 48, 34),
                (154, 118, 78),
                (40, 30, 24),
            ),
        ),
        DemoCase(
            case_id="SYN-003",
            label="Dysplastic nevus",
            suspected_condition="Atypical / dysplastic nevus",
            skin_tone=2,
            features=LesionFeatures(
                asymmetry=0.45,
                border_irregularity=0.50,
                color_variegation=0.52,
                diameter_mm=5.8,
                evolution_score=0.18,
                ulceration=False,
                atypical_network=False,
            ),
            center_xy=(0.42, 0.58),
            axes_frac=(0.15, 0.11),
            angle_deg=35.0,
            palette=((151, 88, 70), (109, 63, 44), (183, 132, 105)),
        ),
        DemoCase(
            case_id="SYN-004",
            label="Suspicious melanoma pattern",
            suspected_condition="Melanoma suspicious pattern",
            skin_tone=5,
            features=LesionFeatures(
                asymmetry=0.82,
                border_irregularity=0.88,
                color_variegation=0.90,
                diameter_mm=9.5,
                evolution_score=0.86,
                ulceration=False,
                atypical_network=True,
            ),
            center_xy=(0.52, 0.48),
            axes_frac=(0.18, 0.14),
            angle_deg=8.0,
            palette=(
                (45, 32, 30),
                (88, 50, 42),
                (28, 24, 32),
                (120, 72, 58),
            ),
        ),
        DemoCase(
            case_id="SYN-005",
            label="Basal cell carcinoma pattern",
            suspected_condition="Basal cell carcinoma pattern",
            skin_tone=2,
            features=LesionFeatures(
                asymmetry=0.42,
                border_irregularity=0.55,
                color_variegation=0.48,
                diameter_mm=7.5,
                evolution_score=0.45,
                ulceration=True,
                atypical_network=False,
            ),
            center_xy=(0.47, 0.54),
            axes_frac=(0.14, 0.12),
            angle_deg=-25.0,
            palette=((196, 164, 150), (151, 86, 76), (220, 198, 186)),
        ),
        DemoCase(
            case_id="SYN-006",
            label="Squamous / keratoacanthoma pattern",
            suspected_condition="Squamous cell carcinoma or keratoacanthoma pattern",
            skin_tone=4,
            features=LesionFeatures(
                asymmetry=0.62,
                border_irregularity=0.70,
                color_variegation=0.66,
                diameter_mm=11.0,
                evolution_score=0.78,
                ulceration=True,
                atypical_network=False,
            ),
            center_xy=(0.58, 0.50),
            axes_frac=(0.15, 0.13),
            angle_deg=50.0,
            palette=((174, 103, 83), (126, 63, 50), (214, 174, 151)),
        ),
    )


def _classify(
    features: LesionFeatures,
    low_threshold: float,
    high_threshold: float,
) -> Prediction:
    """
    Transparent heuristic review-risk score for demo purposes only.

    This is not a trained CNN and is not clinically validated. It exists to
    demonstrate a conservative decision-support output contract.
    """
    diameter_signal = min(
        max((features.diameter_mm - DIAMETER_CENTER_MM) / DIAMETER_SCALE_MM, 0.0),
        1.0,
    )

    logit = (
        MODEL_INTERCEPT
        + MODEL_WEIGHT_ASYM * features.asymmetry
        + MODEL_WEIGHT_BORDER * features.border_irregularity
        + MODEL_WEIGHT_COLOR * features.color_variegation
        + MODEL_WEIGHT_DIAMETER * diameter_signal
        + MODEL_WEIGHT_EVOLUTION * features.evolution_score
        + (MODEL_WEIGHT_ULCERATION if features.ulceration else 0.0)
        + (MODEL_WEIGHT_ATYPICAL_NETWORK if features.atypical_network else 0.0)
    )

    # Numerical safety.
    logit = max(min(logit, 50.0), -50.0)
    probability = 1.0 / (1.0 + math.exp(-logit))
    probability = min(max(probability, 0.0), 1.0)

    if probability < low_threshold:
        category: RiskCategory = "Low"
    elif probability < high_threshold:
        category = "Moderate"
    else:
        category = "High"

    reasons: list[str] = []

    if features.asymmetry >= FEATURE_ALERT_THRESHOLD:
        reasons.append("marked asymmetry")
    if features.border_irregularity >= FEATURE_ALERT_THRESHOLD:
        reasons.append("irregular or notched border")
    if features.color_variegation >= FEATURE_ALERT_THRESHOLD:
        reasons.append("multiple colors")
    if features.diameter_mm >= DIAMETER_ALERT_MM:
        reasons.append(f"diameter >= {DIAMETER_ALERT_MM:.1f} mm")
    if features.evolution_score >= EVOLUTION_ALERT_THRESHOLD:
        reasons.append("reported change or rapid evolution")
    if features.ulceration:
        reasons.append("ulceration or erosion")
    if features.atypical_network:
        reasons.append("atypical dermoscopic network pattern")

    if not reasons:
        reasons.append("no individual high-weight demo signal")

    return Prediction(
        risk_probability=round(probability, 4),
        risk_category=category,
        reasons=tuple(reasons),
    )


def _polygon_points(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    irregularity: float,
    seed: int,
    point_count: int = 72,
) -> list[tuple[int, int]]:
    """
    Generate a closed irregular lesion contour.

    The contour is deterministic for a given seed and is used only for
    synthetic image rendering.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * math.pi, point_count, endpoint=False)

    radius = np.ones_like(theta, dtype=np.float32)
    harmonics = ((1, 0.42), (2, 0.26), (3, 0.16), (5, 0.09))

    for k, amp in harmonics:
        phase = rng.uniform(0.0, 2.0 * math.pi)
        radius += amp * irregularity * np.sin(k * theta + phase)

    radius = np.clip(radius, 0.45, 1.65)

    x = center[0] + axes[0] * radius * np.cos(theta)
    y = center[1] + axes[1] * radius * np.sin(theta)

    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx = x - center[0]
    dy = y - center[1]

    xr = center[0] + dx * cos_a - dy * sin_a
    yr = center[1] + dx * sin_a + dy * cos_a

    return [(int(round(float(px))), int(round(float(py)))) for px, py in zip(xr, yr)]


def _create_static_background(
    width: int,
    height: int,
    skin_tone: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic synthetic skin background."""
    base = np.array(
        SKIN_TONE_COLORS.get(skin_tone, SKIN_TONE_COLORS[3]),
        dtype=np.float32,
    )

    ys = np.linspace(0.93, 1.07, height, dtype=np.float32)[:, None, None]
    xs = np.linspace(0.96, 1.04, width, dtype=np.float32)[None, :, None]

    image = base[None, None, :] * ys * xs

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 3.0, (height, width, 1)).astype(np.float32)
    image = image + noise

    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _attention_mask(
    case: DemoCase,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Simulated attention map.

    This is not true Grad-CAM from a CNN. It is a deterministic visual proxy
    that emphasizes the lesion body and border, suitable only for demo UX.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    cx = case.center_xy[0] * width
    cy = case.center_xy[1] * height
    min_dim = min(width, height)

    ax = max(case.axes_frac[0] * min_dim, 1.0)
    ay = max(case.axes_frac[1] * min_dim, 1.0)

    dx = (xx - cx) / ax
    dy = (yy - cy) / ay
    r = np.sqrt(dx * dx + dy * dy)

    inside = r <= 1.18

    core = np.exp(-(r * r) / (2.0 * 0.52 * 0.52))
    border = np.exp(-((r - 0.88) ** 2) / (2.0 * 0.17 * 0.17))

    mask = inside * (0.38 * core + 0.82 * border)

    rng = np.random.default_rng(_stable_seed(case.case_id + "heatmap"))
    noise_sigma = 0.02 + 0.07 * case.features.color_variegation
    noise = rng.normal(0.0, noise_sigma, mask.shape).astype(np.float32)

    mask = np.clip(mask + noise * inside, 0.0, 1.0)

    max_value = float(mask.max())
    if max_value > 0.0:
        mask = mask / max_value

    return mask


def _apply_heatmap(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.36,
) -> np.ndarray:
    """Overlay a Jet colormap attention mask on an RGB frame."""
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        frame_rgb,
        1.0 - alpha,
        heat_rgb,
        alpha,
        0.0,
    )

    return blended.astype(np.uint8)


def _draw_lesion(
    draw: ImageDraw.ImageDraw,
    case: DemoCase,
    width: int,
    height: int,
    frame_idx: int,
) -> None:
    """Draw a synthetic lesion with mild temporal motion."""
    center = (case.center_xy[0] * width, case.center_xy[1] * height)
    min_dim = min(width, height)

    base_axes = (
        case.axes_frac[0] * min_dim,
        case.axes_frac[1] * min_dim,
    )

    breath = 1.0 + 0.012 * math.sin(frame_idx * 0.22)
    axes = (base_axes[0] * breath, base_axes[1] * breath)

    angle = case.angle_deg + 1.5 * math.sin(frame_idx * 0.13)
    base_seed = _stable_seed(case.case_id)
    irregularity = case.features.border_irregularity

    main_points = _polygon_points(
        center=center,
        axes=axes,
        angle_deg=angle,
        irregularity=irregularity,
        seed=base_seed,
    )
    draw.polygon(main_points, fill=case.palette[0])

    patch_count = int(1 + 5 * case.features.color_variegation)

    for idx in range(patch_count):
        patch_seed = base_seed + 1000 + idx
        prng = np.random.default_rng(patch_seed)

        ox = float(prng.normal(0.0, axes[0] * 0.20))
        oy = float(prng.normal(0.0, axes[1] * 0.20))

        patch_center = (center[0] + ox, center[1] + oy)
        patch_axes = (
            axes[0] * float(prng.uniform(0.16, 0.42)),
            axes[1] * float(prng.uniform(0.16, 0.42)),
        )
        patch_angle = angle + float(prng.uniform(-35.0, 35.0))

        patch_points = _polygon_points(
            center=patch_center,
            axes=patch_axes,
            angle_deg=patch_angle,
            irregularity=max(0.15, irregularity * 0.65),
            seed=patch_seed,
        )

        if len(case.palette) > 1:
            color = case.palette[(idx % (len(case.palette) - 1)) + 1]
        else:
            color = case.palette[0]

        draw.polygon(patch_points, fill=color)

    if case.features.ulceration:
        ulcer_radius = min(axes[0], axes[1]) * 0.22
        bbox = (
            center[0] - ulcer_radius,
            center[1] - ulcer_radius,
            center[0] + ulcer_radius,
            center[1] + ulcer_radius,
        )
        draw.ellipse(bbox, fill=(176, 48, 48))

    if case.features.atypical_network:
        network_rng = np.random.default_rng(base_seed + 77)

        for _ in range(8):
            theta = network_rng.uniform(0.0, 2.0 * math.pi)
            rr = network_rng.uniform(0.62, 0.98)

            x1 = center[0] + axes[0] * rr * math.cos(theta)
            y1 = center[1] + axes[1] * rr * math.sin(theta)

            x2 = center[0] + axes[0] * rr * math.cos(theta + 0.35)
            y2 = center[1] + axes[1] * rr * math.sin(theta + 0.35)

            draw.line((x1, y1, x2, y2), fill=(38, 28, 32), width=2)


def _draw_ui_overlay(
    draw: ImageDraw.ImageDraw,
    case: DemoCase,
    prediction: Prediction,
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Draw clinician-facing demo overlay text."""
    draw.rectangle((0, 0, width, 58), fill=(16, 20, 26))

    category_color = {
        "Low": (72, 196, 112),
        "Moderate": (242, 184, 56),
        "High": (232, 76, 76),
    }.get(prediction.risk_category, (220, 220, 220))

    draw.text(
        (12, 9),
        f"SIMULATED LIVE DEMO | {case.label}",
        fill=(238, 242, 246),
    )

    draw.text(
        (12, 31),
        (
            f"Review-risk probability: {prediction.risk_probability:.2f} | "
            f"Action: {prediction.risk_category} | "
            f"Low/High gates: {low_threshold:.2f}/{high_threshold:.2f}"
        ),
        fill=category_color,
    )

    draw.rectangle((0, height - 64, width, height), fill=(16, 20, 26))

    draw.text(
        (12, height - 52),
        (
            f"{case.case_id} | tone {case.skin_tone} | "
            f"frame {frame_idx + 1}/{total_frames} | "
            "Decision support only; clinician owns final decision"
        ),
        fill=(214, 220, 226),
    )

    top_reason = prediction.reasons[0] if prediction.reasons else "No signal"
    draw.text(
        (12, height - 28),
        f"Top simulated signal: {top_reason}",
        fill=(245, 245, 245),
    )


def _render_frame(
    case: DemoCase,
    prediction: Prediction,
    background: np.ndarray,
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    show_heatmap: bool,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Render one RGB frame for the simulated live demo."""
    frame = background.copy()

    rng = np.random.default_rng(_stable_seed(case.case_id) + frame_idx)

    flicker = float(rng.normal(1.0, 0.004))
    frame = np.clip(frame.astype(np.float32) * flicker, 0.0, 255.0).astype(np.uint8)

    fine_noise = rng.normal(0.0, 1.5, (height, width, 1)).astype(np.float32)
    frame = np.clip(frame.astype(np.float32) + fine_noise, 0.0, 255.0).astype(np.uint8)

    pil_image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    _draw_lesion(draw, case, width, height, frame_idx)

    frame = np.array(pil_image, dtype=np.uint8)

    if show_heatmap:
        mask = _attention_mask(case, width, height)
        frame = _apply_heatmap(frame, mask)

    if total_frames > 1:
        scan_y = int((frame_idx / (total_frames - 1)) * (height - 3))
        frame[scan_y : scan_y + 3, :] = np.clip(
            frame[scan_y : scan_y + 3, :].astype(np.int16) + 55,
            0,
            255,
        ).astype(np.uint8)

    pil_image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    _draw_ui_overlay(
        draw=draw,
        case=case,
        prediction=prediction,
        width=width,
        height=height,
        frame_idx=frame_idx,
        total_frames=total_frames,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    return np.array(pil_image, dtype=np.uint8)


def _iter_demo_frames(
    cases: Sequence[DemoCase],
    low_threshold: float,
    high_threshold: float,
    show_heatmap: bool,
    frames_per_case: int,
    width: int,
    height: int,
) -> Iterable[np.ndarray]:
    """Yield RGB frames for all selected demo cases."""
    for case in cases:
        prediction = _classify(case.features, low_threshold, high_threshold)
        background = _create_static_background(
            width=width,
            height=height,
            skin_tone=case.skin_tone,
            seed=_stable_seed(case.case_id),
        )

        for frame_idx in range(frames_per_case):
            yield _render_frame(
                case=case,
                prediction=prediction,
                background=background,
                width=width,
                height=height,
                frame_idx=frame_idx,
                total_frames=frames_per_case,
                show_heatmap=show_heatmap,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )


@st.cache_data(show_spinner=False)
def _generate_demo_video(
    case_ids: tuple[str, ...],
    show_heatmap: bool,
    low_threshold: float,
    high_threshold: float,
    fps: int,
    total_frames: int,
    width: int,
    height: int,
) -> tuple[bytes, str, str]:
    """
    Generate a browser-playable demo video.

    Preferred path:
    1. imageio + libx264 MP4, if available.
    2. OpenCV VideoWriter fallback using available codecs.

    Returns:
        video_bytes, mime_type, file_extension
    """
    cases_by_id = {case.case_id: case for case in _make_demo_cases()}
    selected = [cases_by_id[cid] for cid in case_ids if cid in cases_by_id]

    if not selected:
        return b"", "video/mp4", ".mp4"

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR
    except OSError:
        output_dir = Path(tempfile.mkdtemp(prefix="skin_lesion_demo_"))

    fingerprint_payload = "|".join(
        [
            str(case_ids),
            str(show_heatmap),
            f"{low_threshold:.4f}",
            f"{high_threshold:.4f}",
            str(fps),
            str(total_frames),
            str(width),
            str(height),
        ]
    )

    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    base_name = f"skin_lesion_demo_{fingerprint}"

    frames_per_case = max(6, int(math.ceil(total_frames / len(selected))))

    # -------------------------------------------------------------------------
    # Preferred writer: imageio + H.264, if available.
    # -------------------------------------------------------------------------
    if imageio is not None:
        imageio_path = output_dir / f"{base_name}_imageio.mp4"

        try:
            try:
                writer = imageio.get_writer(
                    str(imageio_path),
                    fps=float(fps),
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )
            except TypeError:
                writer = imageio.get_writer(
                    str(imageio_path),
                    fps=float(fps),
                    codec="libx264",
                )

            for frame_rgb in _iter_demo_frames(
                cases=selected,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                show_heatmap=show_heatmap,
                frames_per_case=frames_per_case,
                width=width,
                height=height,
            ):
                writer.append_data(frame_rgb)

            writer.close()

            if imageio_path.exists() and imageio_path.stat().st_size > 0:
                return imageio_path.read_bytes(), "video/mp4", ".mp4"

        except Exception:
            try:
                if imageio_path.exists():
                    imageio_path.unlink()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Fallback writer: OpenCV VideoWriter.
    # -------------------------------------------------------------------------
    last_error = None

    for fourcc, ext, mime in VIDEO_WRITER_CANDIDATES:
        candidate_path = output_dir / f"{base_name}{ext}"
        writer = None

        try:
            writer = cv2.VideoWriter(
                str(candidate_path),
                cv2.VideoWriter_fourcc(*fourcc),
                float(fps),
                (int(width), int(height)),
            )

            if not writer.isOpened():
                raise RuntimeError(f"OpenCV VideoWriter could not open codec {fourcc}")

            for frame_rgb in _iter_demo_frames(
                cases=selected,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                show_heatmap=show_heatmap,
                frames_per_case=frames_per_case,
                width=width,
                height=height,
            ):
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            writer = None

            if candidate_path.exists() and candidate_path.stat().st_size > 0:
                return candidate_path.read_bytes(), mime, ext

        except Exception as exc:
            last_error = exc
            try:
                if candidate_path.exists() and candidate_path.stat().st_size == 0:
                    candidate_path.unlink()
            except Exception:
                pass

        finally:
            if writer is not None:
                writer.release()

    raise RuntimeError(
        "Could not generate a browser-playable demo video. "
        "Install imageio-ffmpeg or use an OpenCV build with VP8/VP9/H.264 support."
    ) from last_error


def _safe_thresholds(low: float, high: float) -> tuple[float, float, bool]:
    """Ensure high threshold is strictly greater than low threshold."""
    adjusted = False
    if low >= high:
        high = min(1.0, low + 0.05)
        adjusted = True
    return float(round(low, 4)), float(round(high, 4)), adjusted


def main() -> None:
    st.set_page_config(
        page_title="Skin Lesion Classification Assistant",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Skin Lesion Classification Assistant")
    st.caption(
        "Simulated clinical decision-support demo. Uses synthetic images and a "
        "transparent heuristic review-risk score. Not a medical device, not a "
        "diagnosis, and not trained on real patient data."
    )

    cases = _make_demo_cases()
    cases_by_id = {case.case_id: case for case in cases}

    case_options = {
        f"{case.case_id} — {case.label}": case.case_id
        for case in cases
    }

    with st.sidebar:
        st.header("Simulation controls")

        selected_keys = st.multiselect(
            "Demo cases",
            options=list(case_options.keys()),
            default=list(case_options.keys()),
            help=(
                "Select synthetic cases to include in the live demo video. "
                "The default selection covers multiple lesion patterns and "
                "skin-tone bands."
            ),
        )

        show_heatmap = st.checkbox(
            "Show simulated Grad-CAM overlay",
            value=True,
            help=(
                "This is a synthetic attention overlay for demonstration. "
                "It is not a true CNN Grad-CAM explanation."
            ),
        )

        low_input = st.slider(
            "Low review-risk gate",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_LOW_THRESHOLD,
            step=0.01,
        )

        high_input = st.slider(
            "High review-risk gate",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_HIGH_THRESHOLD,
            step=0.01,
        )

        low_threshold, high_threshold, adjusted = _safe_thresholds(
            low_input,
            high_input,
        )

        if adjusted:
            st.warning("High gate was adjusted to remain above the low gate.")

        fps = st.slider("Video FPS", min_value=5, max_value=30, value=DEFAULT_FPS)
        total_frames = st.slider(
            "Total simulated frames",
            min_value=24,
            max_value=180,
            value=DEFAULT_FRAME_COUNT,
        )

        width = st.slider(
            "Frame width",
            min_value=480,
            max_value=1280,
            value=DEFAULT_WIDTH,
            step=32,
        )

        height = st.slider(
            "Frame height",
            min_value=360,
            max_value=720,
            value=DEFAULT_HEIGHT,
            step=32,
        )

        auto_refresh = st.checkbox(
            "Auto-regenerate video when settings change",
            value=True,
        )

        regenerate = st.button(
            "Regenerate live demo video now",
            type="primary",
        )

        st.divider()
        st.subheader("Governance notes")
        st.markdown(
            """
            - Synthetic data only.
            - Clinician remains accountable for the final decision.
            - Output is assistive review-risk triage, not diagnosis.
            - Subgroup performance must be validated before clinical use.
            """
        )

    selected_ids = tuple(case_options[key] for key in selected_keys)

    if not selected_ids:
        st.warning("Select at least one demo case to generate the simulation.")
        st.stop()

    params_key = (
        selected_ids,
        bool(show_heatmap),
        float(low_threshold),
        float(high_threshold),
        int(fps),
        int(total_frames),
        int(width),
        int(height),
    )

    if "video_params" not in st.session_state:
        st.session_state.video_params = None
        st.session_state.video_payload = None

    should_generate = (
        regenerate
        or st.session_state.video_payload is None
        or (auto_refresh and st.session_state.video_params != params_key)
    )

    if should_generate:
        with st.spinner("Generating simulated live demo video..."):
            try:
                payload = _generate_demo_video(*params_key)
                st.session_state.video_payload = payload
                st.session_state.video_params = params_key
            except Exception as exc:
                st.error(f"Video generation failed: {exc}")
                st.stop()

    video_bytes, video_mime, video_ext = st.session_state.video_payload

    if not video_bytes:
        st.warning("No video was generated. Select at least one demo case.")
        st.stop()

    st.subheader("Live simulated demo")
    st.video(
        video_bytes,
        format=video_mime,
        autoplay=True,
        loop=True,
        muted=True,
    )

    st.download_button(
        "Download generated demo video",
        data=video_bytes,
        file_name=f"skin_lesion_classification_assistant_demo{video_ext}",
        mime=video_mime,
    )

    st.subheader("Case explorer and output contract")

    tabs = st.tabs(["Predictions", "Synthetic data quality", "Example output JSON"])

    with tabs[0]:
        rows: list[dict[str, object]] = []

        for case_id in selected_ids:
            case = cases_by_id[case_id]
            prediction = _classify(case.features, low_threshold, high_threshold)

            rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "skin_tone": case.skin_tone,
                    "diameter_mm": case.features.diameter_mm,
                    "asymmetry": case.features.asymmetry,
                    "border_irregularity": case.features.border_irregularity,
                    "color_variegation": case.features.color_variegation,
                    "evolution_score": case.features.evolution_score,
                    "ulceration": case.features.ulceration,
                    "atypical_network": case.features.atypical_network,
                    "review_risk_probability": prediction.risk_probability,
                    "risk_category": prediction.risk_category,
                    "top_reason": prediction.reasons[0] if prediction.reasons else "",
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Suggested clinician-facing message")

        for row in rows:
            st.write(
                f"**{row['case_id']}**: {row['risk_category']} review-risk. "
                f"{row['top_reason']}. This assistive signal does not replace "
                "examination, dermoscopy, biopsy, or specialist judgment."
            )

    with tabs[1]:
        all_rows = [
            {
                "case_id": case.case_id,
                "label": case.label,
                "suspected_condition": case.suspected_condition,
                "skin_tone": case.skin_tone,
            }
            for case in cases
        ]

        all_df = pd.DataFrame(all_rows)
        st.dataframe(all_df, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Synthetic cases", len(all_df))
        col2.metric("Distinct labels", int(all_df["label"].nunique()))
        col3.metric("Skin-tone bands represented", int(all_df["skin_tone"].nunique()))
        col4.metric("Real patient records", 0)

        st.caption(
            "Balance check: this demo intentionally covers multiple lesion "
            "appearances and skin-tone bands, but it is not epidemiologically "
            "representative. A production dataset must be audited for class "
            "prevalence, anatomical site, device/camera mix, consent basis, "
            "licensing, label agreement, and subgroup performance."
        )

    with tabs[2]:
        first_case = cases_by_id[selected_ids[0]]
        first_prediction = _classify(
            first_case.features,
            low_threshold,
            high_threshold,
        )

        output_example = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "model_version": MODEL_VERSION,
            "input_id": first_case.case_id,
            "modality": "clinical_photo_simulated",
            "body_site": "simulated_trunk",
            "laterality": "not_applicable_synthetic",
            "risk_probability": first_prediction.risk_probability,
            "risk_category": first_prediction.risk_category,
            "thresholds": {
                "low": low_threshold,
                "high": high_threshold,
            },
            "evidence": {
                "simulated_grad_cam_enabled": bool(show_heatmap),
                "top_signals": list(first_prediction.reasons),
            },
            "human_factors": {
                "decision_support_only": True,
                "clinician_accountable": True,
                "fallback_action": "Refer to dermatology or manage according to local standard of care.",
            },
            "disclaimer": (
                "Decision support only. Not a diagnosis. Clinician remains "
                "responsible for final patient management."
            ),
        }

        st.json(output_example)


if __name__ == "__main__":
    main()