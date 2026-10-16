from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
IMAGE_DIR = ROOT / "starter_pack" / "images" / "part2_frequencies"
OUTPUT_DIR = Path(".")

from starter_pack.code.align_images import align_images
from starter_pack.code.utils import load_image


# =====================================================================
# Filter & Two-Band Blending Functions
#
# Low frequencies blended with smoothed mask (smooth color/lighting):
#   I_low_blended = I_apple_low * M_blur + I_orange_low * (1 - M_blur)
#
# High frequencies blended with sharp mask (crisp texture preservation):
#   I_high_blended = I_apple_high * M + I_orange_high * (1 - M)
# =====================================================================

def gaussian_kernel(sigma, radius=None):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve_1d_reflect(image, kernel, axis):
    radius = len(kernel) // 2
    if axis == 0:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="reflect")
        result = np.empty_like(image, dtype=np.float32)
        for row in range(image.shape[0]):
            result[row] = np.sum(padded[row:row + len(kernel)] * kernel[:, None], axis=0)
    elif axis == 1:
        padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
        result = np.empty_like(image, dtype=np.float32)
        for column in range(image.shape[1]):
            result[:, column] = np.sum(padded[:, column:column + len(kernel)] * kernel[None, :], axis=1)
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    return result


def convolve_reflect(image, kernel):
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")
    horizontal = convolve_1d_reflect(image, kernel, axis=1)
    return convolve_1d_reflect(horizontal, kernel, axis=0)


def gaussian_blur(image, sigma):
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}")
    kernel = gaussian_kernel(sigma)
    if image.ndim == 2:
        return convolve_reflect(image, kernel)
    channels = [convolve_reflect(image[:, :, channel], kernel) for channel in range(image.shape[2])]
    return np.stack(channels, axis=2)


def blend_two_band(image_left, image_right, mask_left, sigma):
    """Blends two images using two-band frequency blending."""
    low_left = gaussian_blur(image_left, sigma)
    high_left = image_left - low_left

    low_right = gaussian_blur(image_right, sigma)
    high_right = image_right - low_right

    mask_blurred = gaussian_blur(mask_left, sigma)

    low_blended = low_left * mask_blurred + low_right * (1.0 - mask_blurred)
    high_blended = high_left * mask_left + high_right * (1.0 - mask_left)

    return low_blended + high_blended


def display_masks(mask_apple, mask_orange, apple, orange_aligned):
    """Displays the binary masks and isolated sides before applying the blend."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(mask_apple, cmap="gray")
    axes[0].set_title("Apple Mask (Left Half = 1)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(apple * mask_apple, 0, 1))
    axes[1].set_title("Masked Apple (Left Only)")
    axes[1].axis("off")

    axes[2].imshow(mask_orange, cmap="gray")
    axes[2].set_title("Orange Mask (Right Half = 1)")
    axes[2].axis("off")

    axes[3].imshow(np.clip(orange_aligned * mask_orange, 0, 1))
    axes[3].set_title("Masked Orange (Right Only)")
    axes[3].axis("off")

    fig.suptitle("Spatial Masks & Masked Inputs Before Blending", fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_blends(name, image_left, image_right, mask_left, sigmas):
    """Compares masked two-band blends across different seam sigma widths."""
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.0 * len(sigmas), 3.5))
    if len(sigmas) == 1:
        axes = [axes]

    for ax, s in zip(axes, sigmas):
        print(f"[{name}] Blending seam with σ = {s}...")
        blended = blend_two_band(image_left, image_right, mask_left, sigma=s)
        ax.imshow(np.clip(blended, 0, 1))
        ax.set_title(f"Seam σ = {s}")
        ax.axis("off")

    fig.suptitle(f"{name} (Two-Band Seam Tuning)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ==========================================
# 1. Load and Align Images
# ==========================================

apple = load_image(IMAGE_DIR / "apple.jpg")
orange = load_image(IMAGE_DIR / "orange.jpg")

# Starter pack pairs are pre-aligned (pts1 == pts2 represents identity mapping)
apple_points = [(256, 145), (256, 380)]
orange_points = [(256, 145), (256, 380)]
orange_aligned = align_images(apple, orange, apple_points, orange_points)


# ==========================================
# 2. Construct and Show Masks Before Blending
# ==========================================

h_apple, w_apple = apple.shape[:2]
mid_x = w_apple // 2

# Left mask for apple (1 on left half, 0 on right half)
mask_apple = np.zeros_like(apple)
mask_apple[:, :mid_x] = 1.0

# Right mask for orange (0 on left half, 1 on right half)
mask_orange = 1.0 - mask_apple

# Visualize the masks and isolated halves before merging
display_masks(mask_apple, mask_orange, apple, orange_aligned)


# ==========================================
# 3. Two-Band Seam Tuning Sweep
# ==========================================

# NOTE ON SIGMA SELECTION:
# We tune sigma across [2, 5, 10, 20, 40] to find the best seam width:
# - Low sigma (e.g., 2) leaves a harsh, unnatural cut line down the center.
# - High sigma (e.g., 40) causes color bleeding across the fruit boundaries.
# - Mid-range (10 to 20) smoothly blends the red-to-orange gradient while keeping fruit textures clean.
sigmas_apple_orange = [2, 5, 10, 20, 40]
compare_blends("Apple (Left) + Orange (Right) Blend", apple, orange_aligned, mask_apple, sigmas_apple_orange)


# ==========================================
# 4. Final Blend Display (sigma = 15)
# ==========================================

best_seam_sigma = 15
final_blend = blend_two_band(apple, orange_aligned, mask_apple, sigma=best_seam_sigma)

plt.figure(figsize=(6, 6))
plt.imshow(np.clip(final_blend, 0, 1))
plt.title(f"Final Apple + Orange Blend (Oraple) with Seam σ = {best_seam_sigma}")
plt.axis("off")
plt.tight_layout()
plt.show()

assert final_blend.shape == apple.shape
assert np.isfinite(final_blend).all()
print("Apple + Orange blend completed successfully.")