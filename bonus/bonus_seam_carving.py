

import argparse
import os
import sys
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
# ----------------------------------------------------------------------------
# Energy: gradient magnitude of 1.2 (D_x = [1 0 -1], D_y = D_x^T), computed with
# our own separable convolution (a 1-D filter along one axis, reflect padding).
# ----------------------------------------------------------------------------
def conv1d(im, k, axis):
    """1-D convolution of a 2-D array along `axis`, same size, reflect padding.
    Convolution = correlation with the flipped filter."""
    k = np.asarray(k, dtype=np.float64)[::-1]
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(im, pad, mode="reflect")
    out = np.zeros_like(im, dtype=np.float64)
    n = im.shape[axis]
    for i, w in enumerate(k):
        if axis == 1:
            out += w * p[:, i:i + n]
        else:
            out += w * p[i:i + n, :]
    return out
 
 
def to_gray(img):
    return img @ np.array([0.299, 0.587, 0.114])
 
 
def energy(img):
    g = to_gray(img)
    dx = conv1d(g, [1, 0, -1], axis=1)
    dy = conv1d(g, [1, 0, -1], axis=0)
    return np.sqrt(dx ** 2 + dy ** 2)  # float64
 
 
# ----------------------------------------------------------------------------
# Dynamic program
# ----------------------------------------------------------------------------
def find_seam(E):
    """Vertical seam of minimum total energy. Returns column index per row."""
    h, w = E.shape
    M = E.astype(np.float64).copy()
    for i in range(1, h):
        prev = M[i - 1]
        left = np.empty(w); left[0] = np.inf; left[1:] = prev[:-1]
        right = np.empty(w); right[-1] = np.inf; right[:-1] = prev[1:]
        M[i] += np.minimum(np.minimum(left, prev), right)
    seam = np.empty(h, dtype=np.int64)
    seam[-1] = np.argmin(M[-1])
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo, hi = max(j - 1, 0), min(j + 2, w)
        seam[i] = lo + np.argmin(M[i, lo:hi])
    return seam
 
 
def remove_seam(a, seam):
    """Remove one pixel per row from a (h, w) or (h, w, c) array."""
    h, w = a.shape[:2]
    keep = np.ones((h, w), dtype=bool)
    keep[np.arange(h), seam] = False
    return a[keep].reshape((h, w - 1) + a.shape[2:])
 
 
# ----------------------------------------------------------------------------
# Step 1: object removal
# ----------------------------------------------------------------------------
def remove_object(img, mask):
    img = img.copy()
    mask = mask.copy().astype(bool)
    k = 0
    while mask.any():
        if img.shape[1] <= 2:
            raise RuntimeError("Mask too large: image would vanish.")
        E = energy(img)
        E[mask] = -1e6
        seam = find_seam(E)
        img = remove_seam(img, seam)
        mask = remove_seam(mask, seam)
        k += 1
    return img, k
 
 
# ----------------------------------------------------------------------------
# Step 2: find k seams on a copy, in the coordinates of the reduced photo
# ----------------------------------------------------------------------------
def find_k_seams(img, k):
    h, w = img.shape[:2]
    copy = img.copy()
    cols = np.tile(np.arange(w), (h, 1))        # column in the reduced photo
    is_seam = np.zeros((h, w), dtype=bool)
    rows = np.arange(h)
    for _ in range(k):
        seam = find_seam(energy(copy))
        is_seam[rows, cols[rows, seam]] = True  # mark where it sat in the reduced photo
        copy = remove_seam(copy, seam)
        cols = remove_seam(cols, seam)
    return is_seam
 
 
# ----------------------------------------------------------------------------
# Step 3: widen row by row
# ----------------------------------------------------------------------------
def insert_seams(img, is_seam):
    h, w, c = img.shape
    left = np.concatenate([img[:, :1], img[:, :-1]], axis=1)    # border: itself
    right = np.concatenate([img[:, 1:], img[:, -1:]], axis=1)
    first = (img + left) / 2.0
    second = (img + right) / 2.0
    n = int(is_seam[0].sum())
    out = np.empty((h, w + n, c))
    inserted = np.zeros((h, w + n), dtype=bool)
    for r in range(h):
        s = is_seam[r]
        counts = 1 + s.astype(np.int64)
        pos = np.cumsum(counts) - counts                       # start of each pixel's output
        out[r, pos] = np.where(s[:, None], first[r], img[r])
        out[r, pos[s] + 1] = second[r][s]
        inserted[r, pos[s]] = True
        inserted[r, pos[s] + 1] = True
    return out, inserted
 
 
# ----------------------------------------------------------------------------
# Masks and I/O
# ----------------------------------------------------------------------------
def load_image(path, max_side=1000):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = max(rgb.shape[:2])
    if s > max_side:
        f = max_side / s
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float64) / 255.0
 
 
def build_mask(args, shape):
    h, w = shape[:2]
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            sys.exit(f"Cannot read mask: {args.mask}")
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m > 127
    m = np.zeros((h, w), np.uint8)
    if args.ellipse:
        cx, cy, rx, ry = [int(round(v)) for v in args.ellipse]
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    elif args.polygon:
        pts = np.array([[int(v) for v in p.split(",")] for p in args.polygon], np.int32)
        cv2.fillPoly(m, [pts], 255)
    elif args.roi:
        bgr = cv2.cvtColor((np.clip(args.img_for_roi, 0, 1) * 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        x, y, rw, rh = cv2.selectROI("Drag a box over the object, then press ENTER",
                                     bgr, showCrosshair=False)
        cv2.destroyAllWindows()
        if rw == 0 or rh == 0:
            sys.exit("No box selected.")
        m[y:y + rh, x:x + rw] = 255
    else:
        sys.exit("Give --mask, --ellipse, --polygon or --roi.")
    return m > 0
 
 
def grow_mask(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0
 
 
def overlay(img, mask, color, alpha=1.0):
    out = img.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return out
 
 
def save(path, img):
    cv2.imwrite(path, cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8),
                                   cv2.COLOR_RGB2BGR))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--mask", help="mask image, white = object to remove")
    ap.add_argument("--ellipse", nargs=4, type=float, metavar=("CX", "CY", "RX", "RY"))
    ap.add_argument("--polygon", nargs="+", metavar="X,Y")
    ap.add_argument("--roi", action="store_true", help="drag a box over the object")
    ap.add_argument("--grow", type=int, default=3, help="grow the mask by this many pixels")
    ap.add_argument("--out", default="bonus_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
 
    img = load_image(args.photo)
    H, W = img.shape[:2]
    args.img_for_roi = img
    mask = grow_mask(build_mask(args, img.shape), args.grow)
    if not mask.any():
        sys.exit("The mask is empty (nothing to remove).")
 
    # 1. remove the object
    reduced, k = remove_object(img, mask)
    print(f"original width {W}, removed k = {k} seams, reduced width {reduced.shape[1]}")
 
    # 2. + 3. grow back by k seams found on a copy
    is_seam = find_k_seams(reduced, k)
    restored, inserted = insert_seams(reduced, is_seam)
    assert restored.shape[:2] == (H, W), restored.shape
    print(f"restored shape: {restored.shape[:2]} (original {(H, W)})")
 
    with_mask = overlay(img, mask, (1, 0, 0))
    restored_seams = overlay(restored, inserted, (0, 1, 0))
 
    save(os.path.join(args.out, "1_photo_with_mask.png"), with_mask)
    save(os.path.join(args.out, "2_object_removed.png"), reduced)
    save(os.path.join(args.out, "3_restored_width.png"), restored)
    save(os.path.join(args.out, "3_restored_width_seams.png"), restored_seams)
 
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, im, t in zip(ax, [with_mask, reduced, restored_seams],
                        ["Mask over the object, in red",
                         f"Object gone (width {reduced.shape[1]}, k = {k})",
                         f"Width restored ({W}), new seams in green"]):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t); a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "bonus_figure.png"), dpi=150)
    print(f"saved results to {args.out}/")
 
 
if __name__ == "__main__":
    main()
 

