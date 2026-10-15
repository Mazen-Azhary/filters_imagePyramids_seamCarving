import os
import pathlib
import sys

parent_folder_dir = pathlib.Path(__file__).resolve().parent.parent
utils_dir = parent_folder_dir / "starter_pack" / "code" 
images_dir = parent_folder_dir / "starter_pack" / "images" 
sys.path.append(str(utils_dir))
sys.path.append(str(images_dir))

from utils import load_image, save_image, show
