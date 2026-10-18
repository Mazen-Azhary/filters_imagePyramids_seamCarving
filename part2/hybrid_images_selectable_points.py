from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
IMAGE_DIR = ROOT / "starter_pack" / "images" / "part2_frequencies"
OUTPUT_DIR = ROOT / "part2" / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from starter_pack.code.align_images import align_images, pick_points
from starter_pack.code.utils import load_image, save_image


def rgb_to_gray(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    return (0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]).astype(np.float32)


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


def make_hybrid(low_source, high_source, low_sigma, high_sigma):
    low_frequency = gaussian_blur(low_source, low_sigma)
    high_frequency = high_source - gaussian_blur(high_source, high_sigma)
    hybrid = low_frequency + high_frequency
    return low_frequency, high_frequency, hybrid

def compare_sigmas(name, low_source, high_source, sigmas):
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.0 * len(sigmas), 3.5))
    if len(sigmas) == 1:
        axes = [axes]
    is_gray = (low_source.ndim == 2)
    cmap = "gray" if is_gray else None
    for ax, s in zip(axes, sigmas):
        print(f"[{name}] Testing sigma = {s}...")
        _, _, hybrid = make_hybrid(low_source, high_source, low_sigma=s, high_sigma=s)
        ax.imshow(np.clip(hybrid, 0, 1), cmap=cmap)
        ax.set_title(f"\u03c3 = {s}")
        ax.axis("off")
    fig.suptitle(f"{name} (Multi-Sigma Tuning Sweep)", fontsize=13)
    plt.tight_layout()
    plt.show()


def display_hybrid(name, low_source, high_source, low_frequency, high_frequency, hybrid):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    images = [low_source, high_source, low_frequency, high_frequency + 0.5, hybrid]
    titles = [
        "low-frequency source",
        "high-frequency source",
        "low-pass",
        "high-pass (shifted)",
        "hybrid",
    ]
    is_gray = (low_source.ndim == 2)
    cmap = "gray" if is_gray else None
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.clip(image, 0, 1), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.show()


def find_image(*names):
    for name in names:
        path = IMAGE_DIR / name
        if path.exists():
            return path
    names_text = ", ".join(names)
    raise FileNotFoundError(f"Could not find any of these images in {IMAGE_DIR}: {names_text}")


def choose_points(image, image_name):
    print(f"Click 2 matching features on the {image_name} image, in the same order as the man.")
    points = pick_points(image, n=2)
    if len(points) != 2:
        raise ValueError(
            f"Expected 2 points for {image_name}, but received {len(points)}. "
            "Run the script again and complete both clicks."
        )
    print(f"  {image_name.lower()}_eye_points = {points}")
    return points


def create_hybrid(man, target, target_name, man_points, sigma):
    target_points = choose_points(target, target_name)
    target_aligned = align_images(man, target, man_points, target_points)
    print(f"{target_name.lower()}_aligned: shape={target_aligned.shape}")

    man_gray = rgb_to_gray(man)
    target_aligned_gray = rgb_to_gray(target_aligned)
    compare_sigmas(
        f"Man + {target_name} (Grayscale)",
        man_gray,
        target_aligned_gray,
        [1, 2, 5, 10, 20, 35],
    )

    man_low, target_high, hybrid = make_hybrid(
        man_gray,
        target_aligned_gray,
        low_sigma=sigma,
        high_sigma=sigma,
    )
    display_hybrid(
        f"Man + {target_name} (Grayscale) - Final Result with \u03c3 = {sigma}",
        man_gray,
        target_aligned_gray,
        man_low,
        target_high,
        hybrid,
    )
    output_path = OUTPUT_DIR / f"man_and_{target_name.lower()}_sigma_{sigma}.png"
    save_image(output_path, hybrid)
    print(f"Saved hybrid to {output_path}")
    return man_gray, target_aligned_gray, target_high, hybrid


man = load_image(IMAGE_DIR / "hybrid_man.jpg")
targets = {
    "Cat": load_image(IMAGE_DIR / "hybrid_cat.jpg"),
    "Dog": load_image(find_image("dog.jpeg", "dog.jpg")),
    # "Dog": load_image(find_image("dog2.jpeg", "dog2.jpg")),
    "Eagle": load_image(find_image("eagle.jpeg", "eagle.jpg")),
}

print(f"man: shape={man.shape}, dtype={man.dtype}")
for target_name, target in targets.items():
    print(f"{target_name.lower()}: shape={target.shape}, dtype={target.dtype}")

print("Step 1/4: click 2 points on the MAN image (e.g. left eye, right eye).")
man_eye_points = choose_points(man, "man")
print(f"\nThe next three steps create Man + Cat, Man + Dog, and Man + Eagle.")

best_sigma = 10
results = {}
for target_name, target in targets.items():
    print(f"\n--- Man + {target_name} ---")
    results[target_name] = create_hybrid(
        man,
        target,
        target_name,
        man_eye_points,
        best_sigma,
    )

man_gray, cat_aligned_gray, cat_high, man_cat_hybrid = results["Cat"]
assert np.allclose(
    cat_high,
    cat_aligned_gray - gaussian_blur(cat_aligned_gray, best_sigma),
)
assert gaussian_blur(man_gray, best_sigma).shape == man_gray.shape
assert np.isclose(gaussian_kernel(3).sum(), 1)
constant = np.ones((9, 9), dtype=np.float32)
assert np.allclose(gaussian_blur(constant, 2), constant)
for _, (_, target_gray, _, hybrid) in results.items():
    assert target_gray.shape == man_gray.shape
    assert np.isfinite(hybrid).all()
print("All Man + Cat, Man + Dog, and Man + Eagle checks passed successfully.")