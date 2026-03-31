"""Image augmentation for adverse driving conditions.

Generates augmented versions of Street View frames for training VLMs
on night, fog, rain, and motion blur conditions.
"""

from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple
import cv2
import numpy as np

AugmentationType = Literal["night", "motion_blur", "fog", "rain"]

__all__ = [
    "augment_night",
    "augment_motion_blur",
    "augment_fog",
    "augment_rain",
    "augment_frame",
    "augment_dataset",
]


def augment_night(
    img: np.ndarray,
    intensity: float = 0.7,
    blue_tint: float = 0.15,
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
    result[:, :, 0] += shadow_mask * blue_tint * 50  # B channel
    result[:, :, 1] -= shadow_mask * blue_tint * 20  # G channel
    result[:, :, 2] -= shadow_mask * blue_tint * 30  # R channel

    # Boost bright spots (street lights, signs)
    bright_mask = (gray > 0.7).astype(np.float32)
    bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)

    # Add glow around bright areas
    glow = cv2.GaussianBlur(img.astype(np.float32), (31, 31), 0)
    result = result + glow * bright_mask[:, :, np.newaxis] * 0.3

    # Increase contrast slightly
    result = (result - 128) * 1.1 + 128

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_motion_blur(
    img: np.ndarray,
    kernel_size: int = 15,
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
    result = (img * (1 - mask * 0.7) + result * mask * 0.7).astype(np.uint8)

    return result


def augment_fog(
    img: np.ndarray,
    density: float = 0.5,
    fog_color: Tuple[int, int, int] = (220, 220, 230),
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
    fog_intensity = 1 - np.exp(-density * depth_mask * 3)

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, (h, w))
    fog_intensity = np.clip(fog_intensity + noise, 0, 1)
    fog_intensity = cv2.GaussianBlur(fog_intensity.astype(np.float32), (21, 21), 0)

    # Create fog layer
    fog_layer = np.full_like(img, fog_color, dtype=np.float32)

    # Blend with original
    fog_mask = fog_intensity[:, :, np.newaxis]
    result = img.astype(np.float32) * (1 - fog_mask) + fog_layer * fog_mask

    # Reduce saturation in foggy areas
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - fog_intensity * 0.7)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def augment_rain(
    img: np.ndarray,
    intensity: float = 0.5,
    angle: float = 15,
    streak_length: int = 20,
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
    result = result * (1 - intensity * 0.3)
    result[:, :, 0] += intensity * 15  # Blue tint

    # Create rain streak layer
    rain_layer = np.zeros((h, w), dtype=np.float32)

    # Number of rain streaks based on intensity
    num_streaks = int(intensity * 500)

    for _ in range(num_streaks):
        # Random start position
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        # Calculate streak end point
        length = np.random.randint(streak_length // 2, streak_length)
        angle_rad = np.radians(angle + np.random.uniform(-10, 10))
        x2 = int(x + length * np.sin(angle_rad))
        y2 = int(y + length * np.cos(angle_rad))

        # Draw streak
        brightness = np.random.uniform(0.3, 0.7)
        cv2.line(rain_layer, (x, y), (x2, y2), brightness, 1)

    # Blur rain streaks slightly
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

    # Add rain to image
    rain_color = np.array([200, 200, 210])  # Slight blue-white
    rain_mask = rain_layer[:, :, np.newaxis]
    result = result + rain_mask * rain_color * 0.5

    # Add water droplets on windshield
    num_droplets = int(intensity * 50)
    for _ in range(num_droplets):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.randint(3, 12)

        # Create droplet with refraction effect
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius)

        if y_max > y_min and x_max > x_min:
            # Slight distortion in droplet area
            droplet_region = result[y_min:y_max, x_min:x_max].copy()
            droplet_region = cv2.GaussianBlur(droplet_region, (5, 5), 0)

            # Create circular mask
            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
            local_cx = cx - x_min
            local_cy = cy - y_min
            cv2.circle(mask, (local_cx, local_cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Apply droplet
            mask = mask[:, :, np.newaxis]
            result[y_min:y_max, x_min:x_max] = (
                result[y_min:y_max, x_min:x_max] * (1 - mask * 0.4) +
                droplet_region * mask * 0.4 +
                mask * 30  # Highlight
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
