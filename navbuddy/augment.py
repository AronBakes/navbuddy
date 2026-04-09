"""Image augmentation for adverse driving conditions.

Generates augmented versions of Street View frames for training VLMs
on night, fog, rain, and motion blur conditions.
"""

from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

AugmentationType = Literal["night", "motion_blur", "fog", "rain"]

__all__ = [
    "augment_night",
    "augment_motion_blur",
    "augment_fog",
    "augment_rain",
    "augment_frame",
    "augment_dataset",
]

# ── Night effect constants ───────────────────────────────────────
NIGHT_DARKNESS = 0.7              # Default intensity (0.5-0.9, higher = darker)
NIGHT_BLUE_TINT = 0.15            # Blue shift in shadows (0-0.3)
NIGHT_BLUE_CHANNEL_GAIN = 50      # Blue channel boost in shadow areas
NIGHT_GREEN_CHANNEL_LOSS = 20     # Green channel reduction in shadows
NIGHT_RED_CHANNEL_LOSS = 30       # Red channel reduction in shadows
NIGHT_GLOW_THRESHOLD = 0.7        # Brightness threshold for glow detection
NIGHT_GLOW_KERNEL = (15, 15)      # Gaussian blur kernel for bright spot detection
NIGHT_GLOW_LARGE_KERNEL = (31, 31)  # Larger blur kernel for glow spread
NIGHT_GLOW_INTENSITY = 0.3        # Glow blending strength
NIGHT_CONTRAST_BOOST = 1.1        # Contrast multiplier

# ── Motion blur constants ────────────────────────────────────────
MOTION_BLUR_KERNEL_SIZE = 15      # Default kernel size (odd, 5-25)
MOTION_BLUR_EDGE_BLEND = 0.7     # Edge blur blending factor

# ── Fog effect constants ─────────────────────────────────────────
FOG_DENSITY = 0.5                 # Default fog density (0.2-0.8)
FOG_COLOR_BGR = (220, 220, 230)   # Fog color in BGR
FOG_DEPTH_MULTIPLIER = 3          # Exponential depth scaling
FOG_NOISE_STD = 0.05              # Noise standard deviation for realism
FOG_BLUR_KERNEL = (21, 21)        # Gaussian blur for fog smoothing
FOG_SATURATION_LOSS = 0.7         # Saturation reduction in foggy areas

# ── Rain effect constants ────────────────────────────────────────
RAIN_INTENSITY = 0.5              # Default rain intensity (0.3-0.8)
RAIN_STREAK_ANGLE = 15            # Default streak angle in degrees
RAIN_STREAK_LENGTH = 20           # Default streak length in pixels
RAIN_CONTRAST_LOSS = 0.3          # Contrast reduction factor
RAIN_BLUE_TINT = 15               # Blue channel boost (wet atmosphere)
RAIN_STREAKS_PER_INTENSITY = 500  # Streaks = intensity * this value
RAIN_STREAK_BRIGHTNESS = (0.3, 0.7)  # Random brightness range per streak
RAIN_STREAK_ANGLE_JITTER = 10    # Random angle variation per streak
RAIN_STREAK_BLUR_KERNEL = (3, 3)  # Gaussian blur for streak softening
RAIN_STREAK_COLOR_BGR = (200, 200, 210)  # Streak color (slight blue-white)
RAIN_STREAK_BLEND = 0.5           # Streak blending strength
RAIN_DROPLETS_PER_INTENSITY = 50  # Droplets = intensity * this value
RAIN_DROPLET_RADIUS = (3, 12)     # Random radius range
RAIN_DROPLET_BLUR_KERNEL = (5, 5)  # Gaussian blur for droplet refraction
RAIN_DROPLET_BLEND = 0.4          # Droplet blending strength
RAIN_DROPLET_HIGHLIGHT = 30       # Highlight intensity on droplets


def augment_night(
    img: np.ndarray,
    intensity: float = NIGHT_DARKNESS,
    blue_tint: float = NIGHT_BLUE_TINT,
) -> np.ndarray:
    """Apply night-time effect.

    Args:
        img: Input BGR image
        intensity: Darkness intensity (0.5-0.9, higher = darker)
        blue_tint: Blue color shift for shadows (0-0.3)

    Returns:
        Augmented BGR image
    """
    result = img.astype(np.float32)

    # Reduce overall brightness
    brightness_factor = 1.0 - intensity
    result = result * brightness_factor

    # Add blue tint to darker areas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_mask = 1.0 - gray  # Darker areas get more tint

    # Increase blue channel in shadows
    result[:, :, 0] += shadow_mask * blue_tint * NIGHT_BLUE_CHANNEL_GAIN
    result[:, :, 1] -= shadow_mask * blue_tint * NIGHT_GREEN_CHANNEL_LOSS
    result[:, :, 2] -= shadow_mask * blue_tint * NIGHT_RED_CHANNEL_LOSS

    # Boost bright spots (street lights, signs)
    bright_mask = (gray > NIGHT_GLOW_THRESHOLD).astype(np.float32)
    bright_mask = cv2.GaussianBlur(bright_mask, NIGHT_GLOW_KERNEL, 0)

    # Add glow around bright areas
    glow = cv2.GaussianBlur(img.astype(np.float32), NIGHT_GLOW_LARGE_KERNEL, 0)
    result = result + glow * bright_mask[:, :, np.newaxis] * NIGHT_GLOW_INTENSITY

    # Increase contrast slightly
    result = (result - 128) * NIGHT_CONTRAST_BOOST + 128

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_motion_blur(
    img: np.ndarray,
    kernel_size: int = MOTION_BLUR_KERNEL_SIZE,
    angle: float = 0,
) -> np.ndarray:
    """Apply directional motion blur.

    Args:
        img: Input BGR image
        kernel_size: Blur kernel size (odd number, 5-25)
        angle: Blur direction in degrees (0 = horizontal)

    Returns:
        Augmented BGR image
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Rotate kernel by angle
    center = (kernel_size // 2, kernel_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()  # Normalize

    # Apply blur
    result = cv2.filter2D(img, -1, kernel)

    # Preserve center region sharpness (focal point)
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2

    # Create radial mask - sharp in center, blurred at edges
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    mask = np.clip(dist / max_dist, 0, 1)
    mask = mask[:, :, np.newaxis]

    # Blend original (center) with blurred (edges)
    result = (img * (1 - mask * MOTION_BLUR_EDGE_BLEND) + result * mask * MOTION_BLUR_EDGE_BLEND).astype(np.uint8)

    return result


def augment_fog(
    img: np.ndarray,
    density: float = FOG_DENSITY,
    fog_color: Tuple[int, int, int] = FOG_COLOR_BGR,
) -> np.ndarray:
    """Apply fog/haze effect.

    Args:
        img: Input BGR image
        density: Fog density (0.2-0.8)
        fog_color: BGR color of fog

    Returns:
        Augmented BGR image
    """
    h, w = img.shape[:2]

    # Create depth-based fog mask (bottom = near = clear, top = far = foggy)
    # Simulate perspective depth
    y_coords = np.linspace(0, 1, h)[:, np.newaxis]
    depth_mask = np.broadcast_to(y_coords, (h, w))

    # Apply exponential fog model
    fog_intensity = 1 - np.exp(-density * depth_mask * FOG_DEPTH_MULTIPLIER)

    # Add some noise for realism
    noise = np.random.normal(0, FOG_NOISE_STD, (h, w))
    fog_intensity = np.clip(fog_intensity + noise, 0, 1)
    fog_intensity = cv2.GaussianBlur(fog_intensity.astype(np.float32), FOG_BLUR_KERNEL, 0)

    # Create fog layer
    fog_layer = np.full_like(img, fog_color, dtype=np.float32)

    # Blend with original
    fog_mask = fog_intensity[:, :, np.newaxis]
    result = img.astype(np.float32) * (1 - fog_mask) + fog_layer * fog_mask

    # Reduce saturation in foggy areas
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - fog_intensity * FOG_SATURATION_LOSS)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def augment_rain(
    img: np.ndarray,
    intensity: float = RAIN_INTENSITY,
    angle: float = RAIN_STREAK_ANGLE,
    streak_length: int = RAIN_STREAK_LENGTH,
) -> np.ndarray:
    """Apply rain effect with streaks and reduced visibility.

    Args:
        img: Input BGR image
        intensity: Rain intensity (0.3-0.8)
        angle: Rain streak angle in degrees
        streak_length: Length of rain streaks

    Returns:
        Augmented BGR image
    """
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)

    # Reduce overall contrast and add slight blue tint (wet atmosphere)
    result = result * (1 - intensity * RAIN_CONTRAST_LOSS)
    result[:, :, 0] += intensity * RAIN_BLUE_TINT

    # Create rain streak layer
    rain_layer = np.zeros((h, w), dtype=np.float32)

    # Number of rain streaks based on intensity
    num_streaks = int(intensity * RAIN_STREAKS_PER_INTENSITY)

    for _ in range(num_streaks):
        # Random start position
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        # Calculate streak end point
        length = np.random.randint(streak_length // 2, streak_length)
        angle_rad = np.radians(angle + np.random.uniform(-RAIN_STREAK_ANGLE_JITTER, RAIN_STREAK_ANGLE_JITTER))
        x2 = int(x + length * np.sin(angle_rad))
        y2 = int(y + length * np.cos(angle_rad))

        # Draw streak
        brightness = np.random.uniform(*RAIN_STREAK_BRIGHTNESS)
        cv2.line(rain_layer, (x, y), (x2, y2), brightness, 1)

    # Blur rain streaks slightly
    rain_layer = cv2.GaussianBlur(rain_layer, RAIN_STREAK_BLUR_KERNEL, 0)

    # Add rain to image
    rain_color = np.array(RAIN_STREAK_COLOR_BGR)
    rain_mask = rain_layer[:, :, np.newaxis]
    result = result + rain_mask * rain_color * RAIN_STREAK_BLEND

    # Add water droplets on windshield
    num_droplets = int(intensity * RAIN_DROPLETS_PER_INTENSITY)
    for _ in range(num_droplets):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.randint(*RAIN_DROPLET_RADIUS)

        # Create droplet with refraction effect
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius)

        if y_max > y_min and x_max > x_min:
            # Slight distortion in droplet area
            droplet_region = result[y_min:y_max, x_min:x_max].copy()
            droplet_region = cv2.GaussianBlur(droplet_region, RAIN_DROPLET_BLUR_KERNEL, 0)

            # Create circular mask
            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
            local_cx = cx - x_min
            local_cy = cy - y_min
            cv2.circle(mask, (local_cx, local_cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, RAIN_DROPLET_BLUR_KERNEL, 0)

            # Apply droplet
            mask = mask[:, :, np.newaxis]
            result[y_min:y_max, x_min:x_max] = (
                result[y_min:y_max, x_min:x_max] * (1 - mask * RAIN_DROPLET_BLEND) +
                droplet_region * mask * RAIN_DROPLET_BLEND +
                mask * RAIN_DROPLET_HIGHLIGHT
            )

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_frame(
    img: np.ndarray,
    augmentation: AugmentationType,
    **kwargs,
) -> np.ndarray:
    """Apply specified augmentation to a frame.

    Args:
        img: Input BGR image
        augmentation: Type of augmentation
        **kwargs: Augmentation-specific parameters

    Returns:
        Augmented BGR image
    """
    augment_funcs: Dict[AugmentationType, Callable] = {
        "night": augment_night,
        "motion_blur": augment_motion_blur,
        "fog": augment_fog,
        "rain": augment_rain,

    }

    func = augment_funcs.get(augmentation)
    if func is None:
        raise ValueError(f"Unknown augmentation type: {augmentation}")

    return func(img, **kwargs)


def augment_dataset(
    input_dir: Path,
    output_base: Path,
    augmentations: List[AugmentationType],
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> Dict[str, int]:
    """Augment all frames in a dataset.

    Args:
        input_dir: Directory containing original frames
        output_base: Base output directory (augmented dirs created as siblings)
        augmentations: List of augmentation types to apply
        progress_callback: Optional callback for progress updates
        **kwargs: Augmentation parameters (passed to each augment function)

    Returns:
        Dict mapping augmentation type to number of frames processed
    """
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    # Get all image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")

    results = {}

    for aug_type in augmentations:
        # Create output directory
        output_dir = output_base / f"frames_{aug_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, img_path in enumerate(image_files):
            if progress_callback and i % 10 == 0:
                progress_callback(f"[{aug_type}] Processing {i+1}/{len(image_files)}...")

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Apply augmentation
            augmented = augment_frame(img, aug_type, **kwargs)

            # Save
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), augmented)
            count += 1

        results[aug_type] = count
        if progress_callback:
            progress_callback(f"[{aug_type}] Done: {count} frames")

    return results
