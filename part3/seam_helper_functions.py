import pathlib
import sys
import numpy as np
import cv2 as cv
import matplotlib


parent_folder_dir = pathlib.Path(__file__).resolve().parent.parent
utils_dir = parent_folder_dir / "starter_pack" / "code" 
images_dir = parent_folder_dir / "starter_pack" / "images" / "part3_seams"
sys.path.append(str(utils_dir))
sys.path.append(str(images_dir))
# stting up working directories to make sure code runs from anywhere
 

from utils import load_image, save_image, show
from part1_utils import gradient_maps

def get_image_channels(img):
    """Return the 3 channels of the image """
    channels = []
    if img.ndim == 2:
        return [img]
    elif img.ndim == 3:
        [channels.append(img[..., i]) for i in range(img.shape[2])]
    return np.array(channels)
    

def get_masked_img_channels(img):
    
    img_channels = get_image_channels(rocket_img)
    if len(img_channels) == 3:
        r_img = img_channels[0]
        g_img = img_channels[1]
        b_img = img_channels[2]
        # the picture is in RGB format, so the channels are in the order of R, G, B.
        red_masked_img = np.zeros_like(rocket_img)
        red_masked_img[..., 0] = r_img

        green_masked_img = np.zeros_like(rocket_img)
        green_masked_img[..., 1] = g_img

        blue_masked_img = np.zeros_like(rocket_img)
        blue_masked_img[..., 2] = b_img

        return  [red_masked_img, green_masked_img, blue_masked_img]
    else: 
        print("img size not equal 3 can not get masked images")
        return img_channels


def normalize_image(img):
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if hi == lo:                      # constant image: avoid divide-by-zero
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)

def  get_energy_grad_mag(img):

    img_channels = get_image_channels(img)
    r_img, g_img, b_img = img_channels[0], img_channels[1], img_channels[2]
    Dx = np.array([[1.0, 0.0, -1.0]])  # 1 x 3
    Dy = Dx.T   

    xg, yg, r_gradient_map_rocket_image = gradient_maps(r_img, Dx, Dy)
    xg, yg, g_gradient_map_rocket_image = gradient_maps(g_img, Dx, Dy)
    xg, yg, b_gradient_map_rocket_image = gradient_maps(b_img, Dx, Dy)

    imgs = [r_gradient_map_rocket_image, g_gradient_map_rocket_image, b_gradient_map_rocket_image]
    energy = np.array(sum(im.astype(np.float64) for im in imgs))
    return energy


def _find_vertical_seam(energy):
    """DP: M[i, j] = e[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])."""
    H, W = energy.shape
    M = energy.astype(np.float64).copy()
    for i in range(1, H):
        up = M[i - 1] # the whole row above the current pixel row
        left = np.r_[np.inf, up[:-1]] # left is the row above but shifted to the right, with inf at the first column
        right = np.r_[up[1:], np.inf] # right is the row above but shifted to the left, with inf at the last column
        M[i] += np.minimum(np.minimum(left, up), right) # calculating the minimum energy for the whole row by at each iteration 

    # backtrack from the cheapest pixel in the last row
    seam = np.empty(H, dtype=int)

    # storing vertical seams as a list of column indicies
    seam[-1] = np.argmin(M[-1]) # column which have the lowest energy at the bottom row



    for i in range(H - 2, -1, -1):
        j = seam[i + 1] # getting the column of the seam in the row below
        lo, hi = max(j - 1, 0), min(j + 2, W) # getting the actual cnadidates bounds
        seam[i] = lo + np.argmin(M[i, lo:hi])
    return seam


def _delete(arr, seam):
    """Remove one pixel per row (works for 2D maps and HxWxC images)."""
    H, W = arr.shape[:2]
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False # getting the (row,column) indicies & setting to false for the seam 

    return arr[mask].reshape((H, W - 1) + arr.shape[2:])


def _remove_seams(img, energy, rows, cols, k, energy_fn, removed):
    for _ in range(k):

        print(f" removing seam {_ + 1} of {k}...")
        seam = _find_vertical_seam(energy)
        idx = np.arange(energy.shape[0])
        # record where this seam sat in the ORIGINAL image: (row, col) pairs
        removed.append(np.stack([rows[idx, seam], cols[idx, seam]], axis=1))

        img = _delete(img, seam)
        rows = _delete(rows, seam)
        cols = _delete(cols, seam)
        energy = _delete(energy, seam)
        if energy_fn is not None:          # paper: recompute energy after each removal
            energy = energy_fn(img)
            print(f"energy shape after recomputing: {energy.shape}")
    return img, energy, rows, cols

def _gaussian_blur(g, sigma):
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1)
    k = np.exp(-x**2 / (2 * sigma**2))
    k /= k.sum()

    H, W = g.shape
    p = np.pad(g, r, mode="edge")
    tmp = sum(k[i] * p[:, i:i + W] for i in range(2 * r + 1))   # blur along columns
    return sum(k[i] * tmp[i:i + H] for i in range(2 * r + 1))   # blur along rows


def seam_carve(energy, img, new_size, energy_fn=None):
    """
    energy   : (H, W) energy matrix of img
    img      : (H, W) or (H, W, C) image
    new_size : (new_h, new_w), shrinking only
    energy_fn: optional f(img) -> energy, to recompute energy after each seam

    Returns (removed, resized)
      removed : list of seams in removal order; each is an (L, 2) int array of
                (orig_row, orig_col) for every pixel on that seam
      resized : the new image of shape (new_h, new_w[, C])
    """
    img = np.asarray(img)
    energy = np.asarray(energy, dtype=np.float64)
    H, W = img.shape[:2]
    new_h, new_w = new_size
    if energy.shape != (H, W):
        raise ValueError("energy must have the same H, W as img")
    if not (0 < new_h <= H and 0 < new_w <= W):
        raise ValueError("new_size must be <= current size (shrinking only)")

    # each pixel remembers where it came from; they get cut together with the image
    rows, cols = np.indices((H, W))
    # rows Matrix shape: (H, W), having the row index for all columns in each row
    # cols Matrix shape: (H, W), having the column index for all rows in each column

    # print("rows shape:", rows.shape)
    # print(rows)
    # print("cols shape:", cols.shape)
    # print(cols)
    removed = []

    # 1) vertical seams reduce the width
    img, energy, rows, cols = _remove_seams(
        img, energy, rows, cols, W - new_w, energy_fn, removed)

    # 2) horizontal seams = vertical seams on the transposed image
    img = np.swapaxes(img, 0, 1)
    energy, rows, cols = energy.T, rows.T, cols.T
    img, energy, rows, cols = _remove_seams(
        img, energy, rows, cols, H - new_h, energy_fn, removed)
    img = np.swapaxes(img, 0, 1)

    return removed, img

def draw_removed_seams(removed, img, color=(255, 0, 0)):
    """
    removed : list of (L, 2) int arrays of (orig_row, orig_col), as returned by seam_carve
    img     : the ORIGINAL image, (H, W) or (H, W, C)
    color   : seam color, in the same value range as img (default red for uint8)

    Returns a copy of img with every removed seam painted on it.
    """
    out = np.asarray(img).copy()
    if out.ndim == 2:                          # grayscale -> RGB so the seams are visible
        out = np.stack([out] * 3, axis=2)

    if len(removed) == 0:
        return out

    pts = np.concatenate(removed, axis=0)      # all seams as one (N, 2) array
    out[pts[:, 0], pts[:, 1]] = color          # paint them in one vectorized step
    return out


def resize_subsample(img, new_size):
    """
    Resize by picking pixels (no blending): keep one row/column from each
    successive block of rows/columns.
    img      : (H, W) or (H, W, C)
    new_size : (new_h, new_w), shrinking only
    """
    img = np.asarray(img)
    H, W = img.shape[:2]
    new_h, new_w = new_size

    rows = (np.arange(new_h) * H / new_h).astype(int)   # first row of each block
    cols = (np.arange(new_w) * W / new_w).astype(int)   # first column of each block
    return img[rows][:, cols]

def get_finite_diff_energy(img):
    """Sum over channels of |dI/dx| + |dI/dy|."""
    energy = 0
    for ch in get_image_channels(img):
        gy, gx = np.gradient(ch.astype(np.float64))
        energy = energy + np.abs(gx) + np.abs(gy)
    return energy

def get_dog_energy(img, sigma=1.0, k=1.6):
    """Sum over channels of |G(sigma) * I - G(k*sigma) * I|."""
    energy = 0
    for ch in get_image_channels(img):
        ch = ch.astype(np.float64)
        energy = energy + np.abs(_gaussian_blur(ch, sigma) - _gaussian_blur(ch, k * sigma))
    return energy


if __name__ == "__main__":
    broadway_img = load_image(images_dir / "broadway_tower.png")
    home_img = load_image(images_dir / "home.jpg")
    rocket_img = load_image(images_dir / "rocket.jpg")
    starry_night_img = load_image(images_dir / "starry_night.jpg") 

    curr_img = broadway_img
    # this loads image in rgb format
    Dx = np.array([[1.0, 0.0, -1.0]])  # 1 x 3
    Dy = Dx.T   


    img_channels = get_image_channels(curr_img)
    # print(len(img_channels))
    if len(img_channels) == 3:
        r_img, g_img, b_img = img_channels
        # the picture is in RGB format, so the channels are in the order of R, G, B.
    
        blue_masked_img, green_masked_img, red_masked_img = get_masked_img_channels(curr_img)
        
    else: sys.exit("Image is not a color image with 3 channels.")

    xg, yg, b_gradient_map_rocket_image = gradient_maps(b_img, Dx, Dy)
    xg, yg, g_gradient_map_rocket_image = gradient_maps(g_img, Dx, Dy)
    xg, yg, r_gradient_map_rocket_image = gradient_maps(r_img, Dx, Dy)
    image_energy = get_energy_grad_mag(curr_img)
    # image_energy = curr_img


    normalized_r_gradient = normalize_image(r_gradient_map_rocket_image)
    normalized_g_gradient = normalize_image(g_gradient_map_rocket_image)
    normalized_b_gradient = normalize_image(b_gradient_map_rocket_image)
    # normalized_energy = normalize_image(image_energy)
    normalized_energy = normalize_image(image_energy)

    print("b_gradient_map_rocket_image max pixel:", r_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", g_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", b_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", image_energy.max())
    
    # print(curr_img.shape)    
    # print(image_energy.shape)
    # removed_seams, resized_img = seam_carve(image_energy, curr_img, (curr_img.shape[0]  , curr_img.shape[1] - 200), energy_fn=get_energy_grad_mag)
    # vis = draw_removed_seams(removed_seams, curr_img, color=(255, 0, 0))

    matplotlib.use('TkAgg') # Setting up my backend due to working in a virtual environment.
    # show(*[curr_img, normalized_r_gradient, normalized_g_gradient, normalized_b_gradient, normalized_energy, ], \
    #        titles=["Rocket Energy", "Blue Gradient", "Green Gradient", "Red Gradient", "energy"], size=5)

    # names = ["home.jpg", "rocket.jpg", "starry_night.jpg", "currents.jpg", "mountains.jpg", "anakin.jpeg"]
    names = ["anakin.jpg"]
    for name in names:
        img = load_image(images_dir / name)

        new_dim = (img.shape[0], int(img.shape[1] * 0.7))
        energy = get_finite_diff_energy(img)
        removed_seams, seam_carved_img = seam_carve(energy, img, new_dim, energy_fn=get_energy_grad_mag)
        img_vis = draw_removed_seams(removed_seams, img, color=(255, 0, 0))
        img_resized = resize_subsample(img, new_dim)
        cropped_out = (int(img.shape[0] - new_dim[0]), int(img.shape[1] - new_dim[1]))
        img_cropped = img[cropped_out[0]//2:img.shape[0] - cropped_out[0]//2,
                        cropped_out[1]//2:img.shape[1] - cropped_out[1]//2]

        show(img, img_resized, img_cropped, seam_carved_img,
            titles=["Original", "Resized", "Cropped", "Seam Carved"], size=8)
        show(img, energy, img_vis,
            titles=["Original", "Energy", "Removed Seams"], size=8)
    # show(*[curr_img, resized_img, vis], titles=["Original", "Resized", "Removed Seams"], size=5)
    # show(*[home_img, home_seam_carved_img, home_img_vis], titles=["Original", "Resized", "Removed Seams"], size=5)
    # show(*[rocket_img, rocket_seam_carved_img, rocket_img_vis], titles=["Original", "Resized", "Removed Seams"], size=5)
    # show(*[starry_night_img, starry_night_seam_carved_img, starry_night_img_vis], titles=["Original", "Resized", "Removed Seams"], size=5)
