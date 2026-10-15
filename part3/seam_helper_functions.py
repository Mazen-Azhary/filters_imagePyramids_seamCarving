import pathlib
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg') # Setting up my backend due to working in a virtual environment.


parent_folder_dir = pathlib.Path(__file__).resolve().parent.parent
utils_dir = parent_folder_dir / "starter_pack" / "code" 
images_dir = parent_folder_dir / "starter_pack" / "images" / "part3_seams"
sys.path.append(str(utils_dir))
sys.path.append(str(images_dir))
# stting up working directories to make sure code runs from anywhere
 

from utils import load_image, save_image, show
# from part1_utils import *
rocket_img = load_image(images_dir / "rocket.jpg")
star_img = load_image(images_dir / "starry_night.jpg")
show(*[rocket_img, star_img], titles=["Rocket", "Starry Night"], size=5)
