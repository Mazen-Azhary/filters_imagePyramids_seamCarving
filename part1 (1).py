"""
Assignment 1 - Part 1: Filters and Edges
  1.1 Convolution from scratch (loop / vectorized / separable) + sanity check + timing
  1.2 Finite difference operator
  1.3 Derivative of Gaussian (DoG) filter

Project layout this script expects (part1.py sits NEXT TO the starter_pack folder):

    part1.py
    starter_pack/
        code/utils.py, code/align_images.py
        images/part1_filters/cameraman.png, facade.png

Run from the folder that contains part1.py:
    python part1.py                       # runs 1.1, then 1.2 + 1.3 on facade.png and cameraman.png
    python part1.py --image facade        # only one image (a name from the starter pack, or a file path)
    python part1.py --sigma 2 --rel 0.25  # DoG sigma, and edge threshold = rel * (strong-edge strength)
    python part1.py --thr12 0.2 --thr13 0.08   # or give absolute thresholds yourself
    python part1.py --loop-up-to 31       # also time the (very slow) loop version on the big filters
    python part1.py --show                # open the figures in windows as well

Everything is saved in ./part1_output/
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

# Use a non-interactive backend unless --show is requested (must happen before utils imports pyplot).
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import cv2
import numpy as np
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter_pack"
sys.path.insert(0, str(STARTER / "code"))
from utils import load_image, save_image  # noqa: E402  (starter-pack helper)

import matplotlib.pyplot as plt  # noqa: E402

IMG_DIR = STARTER / "images" / "part1_filters"
OUT_DIR = ROOT / "part1_output"
SHOW = False


# =====================================================================
# 1.1  CONVOLUTION FROM SCRATCH
# =====================================================================
def _pad(img, ph, pw, mode):
    """Pad by ph rows (top and bottom) and pw columns (left and right).
    mode='constant' -> zeros (Part 1).  mode='reflect' -> from Part 2 on."""
    return np.pad(img, ((ph, ph), (pw, pw)), mode=mode)


def conv2d_loop(img, kernel, pad_mode="constant"):
    """Version 1: loops over every output pixel AND every filter entry."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    H, W = img.shape
    kh, kw = kernel.shape
    padded = _pad(img, kh // 2, kw // 2, pad_mode)
    kf = kernel[::-1, ::-1]  # convolution = correlation with the flipped filter
    out = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            s = 0.0
            for i in range(kh):
                for j in range(kw):
                    s += padded[y + i, x + j] * kf[i, j]
            out[y, x] = s
    return out


def conv2d_vectorized(img, kernel, pad_mode="constant"):
    """Version 2: no loop over pixels. Loop over the filter entries only and add a
    shifted copy of the whole image, scaled by that filter entry."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    H, W = img.shape
    kh, kw = kernel.shape
    padded = _pad(img, kh // 2, kw // 2, pad_mode)
    kf = kernel[::-1, ::-1]
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * padded[i:i + H, j:j + W]
    return out


def separable_factors(kernel, tol=1e-10):
    """SVD rank-1 test. Returns (column, row) if the kernel is separable up to
    rounding error, otherwise None."""
    kernel = np.asarray(kernel, dtype=np.float64)
    U, S, Vt = np.linalg.svd(kernel)
    if S[0] == 0:  # all-zero kernel
        return np.zeros((kernel.shape[0], 1)), np.zeros((1, kernel.shape[1]))
    if len(S) > 1 and S[1] > tol * S[0]:  # second singular value not ~0 -> rank > 1
        return None
    r = np.sqrt(S[0])
    return (U[:, 0] * r).reshape(-1, 1), (Vt[0] * r).reshape(1, -1)  # kh x 1, 1 x kw


def conv2d_separable(img, kernel, pad_mode="constant"):
    """Version 3: rank-1 kernel -> convolve with the column, then with the row.
    Otherwise fall back to the vectorized version."""
    factors = separable_factors(kernel)
    if factors is None:
        return conv2d_vectorized(img, kernel, pad_mode)
    col, row = factors
    tmp = conv2d_vectorized(img, col, pad_mode)   # vertical pass
    return conv2d_vectorized(tmp, row, pad_mode)  # horizontal pass


def gaussian_kernel(size, sigma):
    """size x size Gaussian, normalized to sum 1."""
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def sanity_check(n_trials=60, seed=0):
    """All three versions must match scipy.signal.convolve2d(mode='same')."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    n_sep = n_non = 0
    for t in range(n_trials):
        H, W = rng.integers(5, 21, size=2)
        kh, kw = rng.choice([1, 3, 5, 7], size=2)  # odd heights and widths
        img = rng.random((H, W))
        if t % 2 == 0:  # separable kernel
            k = np.outer(rng.standard_normal(kh), rng.standard_normal(kw))
            assert separable_factors(k) is not None, "separable kernel was not detected"
            n_sep += 1
        else:  # random kernel (not separable unless it is 1 pixel thin)
            k = rng.standard_normal((kh, kw))
            n_non += 1
        ref = convolve2d(img, k, mode="same")  # zero padding by default
        for name, f in (("loop", conv2d_loop), ("vectorized", conv2d_vectorized),
                        ("separable", conv2d_separable)):
            out = f(img, k)
            err = np.abs(out - ref).max()
            worst = max(worst, err)
            assert out.shape == ref.shape and np.allclose(out, ref, atol=1e-9), f"{name} mismatch (err={err})"
    print(f"[1.1] sanity check passed: {n_trials} random cases ({n_sep} separable, {n_non} general), "
          f"3 versions each, worst abs error vs scipy = {worst:.2e}")


def _best_time(f, img, k, repeats):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(img, k)
        best = min(best, time.perf_counter() - t0)
    return best


def timing_experiment(sizes=(3, 5, 7, 9, 11, 15, 21, 31), loop_up_to=15):
    """Time the three versions (plus scipy as reference) on a 256x256 image."""
    img = np.random.default_rng(1).random((256, 256))
    print("\n[1.1] Timing on a 256x256 image, Gaussian filters (sigma = size/6), seconds")
    print(f"{'size':>7} | {'loop':>10} | {'vectorized':>10} | {'separable':>10} | {'scipy':>10}")
    print("-" * 60)
    rows = []
    for s in sizes:
        k = gaussian_kernel(s, s / 6.0)
        t_loop = _best_time(conv2d_loop, img, k, 1) if s <= loop_up_to else np.nan
        t_vec = _best_time(conv2d_vectorized, img, k, 3)
        t_sep = _best_time(conv2d_separable, img, k, 3)
        t_sci = _best_time(lambda a, b: convolve2d(a, b, mode="same"), img, k, 3)
        rows.append((s, t_loop, t_vec, t_sep, t_sci))
        loop_txt = f"{t_loop:10.4f}" if not np.isnan(t_loop) else f"{'skipped':>10}"
        print(f"{s:>3}x{s:<3} | {loop_txt} | {t_vec:10.4f} | {t_sep:10.4f} | {t_sci:10.4f}")
    rows = np.array(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, name in ((1, "loop"), (2, "vectorized"), (3, "separable"), (4, "scipy")):
        ax.plot(rows[:, 0], rows[:, col], marker="o", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("filter size k (k x k)")
    ax.set_ylabel("time (s), log scale")
    ax.set_title("Convolution time on a 256x256 image")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, "1_1_timing.png")
    return rows


# =====================================================================
# Helpers: loading, display, saving
# =====================================================================
def load_gray(path, max_side=1000):
    """Grayscale float image in [0,1] (via utils.load_image), at most max_side px on the long side."""
    img = load_image(path, gray=True).astype(np.float64)
    h, w = img.shape
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def show_signed(x, pct=99.5):
    """Signed image -> [0,1] with 0 as mid-gray (utils.show clips to [0,1])."""
    m = np.percentile(np.abs(x), pct)
    m = m if m > 0 else 1.0
    return np.clip(0.5 + 0.5 * x / m, 0, 1)


def show_positive(x, pct=99.5):
    m = np.percentile(x, pct)
    m = m if m > 0 else 1.0
    return np.clip(x / m, 0, 1)


def save_fig(fig, name, outdir=None):
    outdir = Path(outdir) if outdir else OUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("   saved", path.relative_to(ROOT))
    if SHOW:
        plt.show()
    plt.close(fig)


# =====================================================================
# 1.2  FINITE DIFFERENCE OPERATOR
# =====================================================================
Dx = np.array([[1.0, 0.0, -1.0]])  # 1 x 3
Dy = Dx.T                          # 3 x 1


def gradient_maps(img, kx, ky):
    """Two derivative images (one convolution each, separable version) and the gradient magnitude."""
    gx = conv2d_separable(img, kx)
    gy = conv2d_separable(img, ky)
    return gx, gy, np.sqrt(gx ** 2 + gy ** 2)


def auto_threshold(mag, margin, rel):
    """Threshold = rel x (strength of the strong edges). Strong edges are measured by the
    99th percentile of the magnitude in the interior only (the image frame is excluded, since the
    zero padding makes the border gradient artificially huge)."""
    interior = mag[margin:-margin, margin:-margin]
    return rel * np.percentile(interior, 99)


def plot_four_maps(gx, gy, mag, edges, titles, name, outdir):
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    for ax, im, t in zip(axs, (show_signed(gx), show_signed(gy), show_positive(mag), edges), titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis("off")
    save_fig(fig, name, outdir)


def plot_corner(edges, name, outdir, size=80):
    size = min(size, *edges.shape)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(edges[:size, :size], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Edges, top-left {size}x{size} corner enlarged")
    ax.axis("off")
    save_fig(fig, name, outdir)


def plot_threshold_sweep(mag, thresholds, name, outdir):
    fig, axs = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4))
    for ax, t in zip(np.atleast_1d(axs), thresholds):
        ax.imshow(mag > t, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"T = {t:.3f}")
        ax.axis("off")
    save_fig(fig, name, outdir)


def save_individual(prefix, outdir, gx, gy, mag, edges):
    """Also save each map as its own PNG (utils.save_image), handy for the report."""
    outdir.mkdir(parents=True, exist_ok=True)
    save_image(outdir / f"{prefix}_dx.png", show_signed(gx))
    save_image(outdir / f"{prefix}_dy.png", show_signed(gy))
    save_image(outdir / f"{prefix}_magnitude.png", show_positive(mag))
    save_image(outdir / f"{prefix}_edges.png", edges)


def part_1_2(img, outdir, thr=None, rel=0.25):
    margin = 3
    gx, gy, mag = gradient_maps(img, Dx, Dy)
    if thr is None:
        thr = auto_threshold(mag, margin, rel)
        how = f"{rel} x 99th percentile of the interior magnitude"
    else:
        how = "given on the command line"
    edges = (mag > thr).astype(np.float64)
    print(f"\n[1.2] Finite differences on a {img.shape[1]}x{img.shape[0]} image")
    print(f"   threshold T = {thr:.4f}  ({how})")
    print(f"   magnitude: max={mag.max():.3f}  interior 99th pct={np.percentile(mag[margin:-margin, margin:-margin], 99):.3f}"
          f"  | edge pixels: {100 * edges.mean():.1f}%")
    plot_four_maps(gx, gy, mag, edges, ["dI/dx", "dI/dy", "Gradient magnitude", f"Edges (T={thr:.3f})"],
                   "1_2_maps.png", outdir)
    plot_corner(edges, "1_2_corner.png", outdir)
    plot_threshold_sweep(mag, [thr / 3, thr / 1.5, thr, thr * 1.5, thr * 3], "1_2_threshold_sweep.png", outdir)
    save_individual("1_2", outdir, gx, gy, mag, edges)
    return edges


# =====================================================================
# 1.3  DERIVATIVE OF GAUSSIAN (DoG)
# =====================================================================
def dog_filters(sigma):
    size = 2 * int(np.ceil(3 * sigma)) + 1  # covers +-3 sigma
    G = gaussian_kernel(size, sigma)
    DoGx = conv2d_separable(G, Dx)  # filter the kernel itself with Dx ...
    DoGy = conv2d_separable(G, Dy)  # ... and with Dy
    return G, DoGx, DoGy


def part_1_3(img, outdir, sigma=2.0, thr=None, rel=0.25, edges12=None):
    G, DoGx, DoGy = dog_filters(sigma)
    margin = G.shape[0] // 2 + 2
    gx, gy, mag = gradient_maps(img, DoGx, DoGy)  # one convolution per DoG filter
    if thr is None:
        thr = auto_threshold(mag, margin, rel)
        how = f"{rel} x 99th percentile of the interior magnitude"
    else:
        how = "given on the command line"
    edges = (mag > thr).astype(np.float64)
    print(f"\n[1.3] DoG, sigma = {sigma}, filter size {G.shape[0]}x{G.shape[1]}")
    print(f"   DoGx rank-1 (separable): {separable_factors(DoGx) is not None}, "
          f"DoGy rank-1: {separable_factors(DoGy) is not None}")
    print(f"   threshold T = {thr:.4f}  ({how})")
    print(f"   magnitude: max={mag.max():.3f}  | edge pixels: {100 * edges.mean():.1f}%")

    # the two DoG filters as images
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for ax, f, t in zip(axs, (DoGx, DoGy), (f"DoG in x, sigma = {sigma:g}", f"DoG in y, sigma = {sigma:g}")):
        ax.imshow(show_signed(f, pct=100), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(t)
        ax.axis("off")
    save_fig(fig, "1_3_dog_filters.png", outdir)

    plot_four_maps(gx, gy, mag, edges,
                   [f"DoG-x (sigma={sigma:g})", f"DoG-y (sigma={sigma:g})", "Gradient magnitude",
                    f"Edges (T={thr:.3f})"], "1_3_maps.png", outdir)
    plot_corner(edges, "1_3_corner.png", outdir)
    plot_threshold_sweep(mag, [thr / 3, thr / 1.5, thr, thr * 1.5, thr * 3], "1_3_threshold_sweep.png", outdir)
    save_individual("1_3", outdir, gx, gy, mag, edges)
    save_image(outdir / "1_3_dog_x_filter.png", cv2.resize(show_signed(DoGx, 100), None, fx=16, fy=16,
                                                           interpolation=cv2.INTER_NEAREST))
    save_image(outdir / "1_3_dog_y_filter.png", cv2.resize(show_signed(DoGy, 100), None, fx=16, fy=16,
                                                           interpolation=cv2.INTER_NEAREST))

    if edges12 is not None:  # side-by-side for the "how and why do the maps differ" question
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, t in zip(axs, (img, edges12, edges), ("Image", "Edges, finite differences (1.2)",
                                                           f"Edges, DoG sigma={sigma:g} (1.3)")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(t)
            ax.axis("off")
        save_fig(fig, "1_2_vs_1_3_edges.png", outdir)

    # check: blur-then-differentiate equals a single DoG convolution (away from the border)
    ref = conv2d_separable(conv2d_separable(img, G), Dx)
    m = G.shape[0]
    print(f"   check (I*G)*Dx vs I*DoGx, max interior difference: {np.abs(ref - gx)[m:-m, m:-m].max():.1e} "
          "(only kernel truncation)")


# =====================================================================
# MAIN
# =====================================================================
def resolve_images(arg):
    if arg is None:
        names = ["facade", "cameraman"]
        return [(n, IMG_DIR / f"{n}.png") for n in names]
    p = Path(arg)
    if p.exists():
        return [(p.stem, p)]
    p2 = IMG_DIR / (arg if Path(arg).suffix else arg + ".png")
    if p2.exists():
        return [(p2.stem, p2)]
    raise FileNotFoundError(f"image not found: {arg} (looked in {IMG_DIR})")


def main():
    global SHOW
    ap = argparse.ArgumentParser(description="Assignment 1, Part 1")
    ap.add_argument("--image", default=None, help="'facade', 'cameraman', or a path (default: both)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma of the DoG filters")
    ap.add_argument("--rel", type=float, default=0.25,
                    help="threshold as a fraction of the strong-edge strength (used when no absolute T is given)")
    ap.add_argument("--thr12", type=float, default=None, help="absolute edge threshold for 1.2")
    ap.add_argument("--thr13", type=float, default=None, help="absolute edge threshold for 1.3")
    ap.add_argument("--loop-up-to", type=int, default=15,
                    help="largest filter size for which the slow loop version is timed")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--show", action="store_true", help="open the figures in windows as well")
    args = ap.parse_args()
    SHOW = args.show

    # ---- 1.1
    sanity_check()
    if not args.skip_timing:
        timing_experiment(loop_up_to=args.loop_up_to)

    # ---- 1.2 and 1.3
    for name, path in resolve_images(args.image):
        img = load_gray(path)
        print(f"\n================ {name} ({path.name}) ================")
        outdir = OUT_DIR / name
        edges12 = part_1_2(img, outdir, thr=args.thr12, rel=args.rel)
        part_1_3(img, outdir, sigma=args.sigma, thr=args.thr13, rel=args.rel, edges12=edges12)

    print(f"\nDone. Everything is in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
