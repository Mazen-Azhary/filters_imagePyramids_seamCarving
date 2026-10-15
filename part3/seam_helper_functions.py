import pathlib
import sys
import numpy as np
import cv2
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

def  get_image_energy(img, energy_function = "gradient magnitude"):

    img_channels = get_image_channels(curr_img)

    r_img, g_img, b_img = img_channels
    Dx = np.array([[1.0, 0.0, -1.0]])  # 1 x 3
    Dy = Dx.T   

    xg, yg, b_gradient_map_rocket_image = gradient_maps(b_img, Dx, Dy)
    xg, yg, g_gradient_map_rocket_image = gradient_maps(g_img, Dx, Dy)
    xg, yg, r_gradient_map_rocket_image = gradient_maps(r_img, Dx, Dy)

    imgs = [r_img, g_img, b_img]
    result = sum(im.astype(np.float64) for im in imgs)
    return sum

# def get_vertical_seams(img)


if __name__ == "__main__":
    rocket_img = load_image(images_dir / "rocket.jpg") 
    broadway_img = load_image(images_dir / "broadway_tower.png")

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
    # image_energy = get_image_energy(curr_img)
    image_energy = curr_img


    print("\n\nwrong herre")
    normalized_r_gradient = normalize_image(r_gradient_map_rocket_image)
    normalized_g_gradient = normalize_image(g_gradient_map_rocket_image)
    normalized_b_gradient = normalize_image(b_gradient_map_rocket_image)
    # normalized_energy = normalize_image(image_energy)
    normalized_energy = normalize_image(image_energy)

    print("b_gradient_map_rocket_image max pixel:", r_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", g_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", b_gradient_map_rocket_image.max())
    print("b_gradient_map_rocket_image max pixel:", image_energy.max())
    
    matplotlib.use('TkAgg') # Setting up my backend due to working in a virtual environment.
    show(*[curr_img, normalized_r_gradient, normalized_g_gradient, normalized_b_gradient, normalized_energy, ], \
           titles=["Rocket Energy", "Blue Gradient", "Green Gradient", "Red Gradient", "energy"], size=5)
    # show(*[curr_img, blue_masked_img, green_masked_img, red_masked_img], titles=["Rocket Energy", "Blue Gradient", "Green Gradient", "Red Gradient"], size=5)
    # shows colored image in rgb format

    