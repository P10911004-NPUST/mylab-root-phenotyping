import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import skimage as sk
from PIL import Image, TiffImagePlugin
from skimage import filters as sk_filters
from skimage import transform as sk_transform
import czifile

from parameters import *

def create_output_folder(input_folder_path: str, mkdir: bool = False):
    path = Path(input_folder_path).resolve()
    if not path.is_dir():
        raise ValueError(f"Input must be an existing folder: {input_folder_path}")
    new_path = path.with_name("OUT_" + path.name)
    if mkdir and not new_path.exists():
        new_path.mkdir(parents=True, exist_ok=True)
        print(f"Create directory: {new_path.as_posix()}")
    return new_path.as_posix()

def get_img_list(input_folder_path:str, suffix: str | None = ".czi"):
    path = Path(input_folder_path).resolve().rglob("*")
    if suffix is None:
        suffix = IMG_TYPE
    img_list = [i.as_posix() for i in path if i.is_file() and i.suffix.lower() in suffix]
    # img_list = [str(i).replace("\\", "/") for i in img_list]
    return img_list

def scale_min_max(arr: np.ndarray, range=(0, 1)):
    arr = arr + 255.0
    vmin = arr.min()
    vmax = arr.max()
    arr = (arr - vmin) / (vmax - vmin)
    arr = arr * (range[1] - range[0]) + range[0]
    return arr

def convert_to_uint8(arr: np.ndarray):
    if arr.ndim < 2:
        return(np.uint8(scale_min_max(arr)))
    elif arr.ndim == 2:
        arr = scale_min_max(arr) * 255.0
        arr = np.uint8(arr)
        return arr
    else:
        arr = arr.astype(np.double)
        gmin = arr.min()
        gmax = arr.max()
        out = np.empty_like(arr, dtype=np.uint8)
        for c in range(arr.shape[-1]):
            tmp_arr = arr[..., c]
            tmp_arr = (tmp_arr - gmin) / (gmax - gmin)
            out[..., c] = np.uint8(tmp_arr * 255.0)
            # out[..., c] = convert_to_uint8(arr[..., c])
        return out

def gray_to_lut(arr: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    # The lower part (blue -> white)
    lut[:128, 0] = np.linspace(0, 255, 128) # R
    lut[:128, 1] = np.linspace(0, 255, 128) # G
    lut[:128, 2] = 255                      # B
    # The upper part (white -> red)
    lut[128:, 0] = 255                      # R
    lut[128:, 1] = np.linspace(255, 0, 128) # G
    lut[128:, 2] = np.linspace(255, 0, 128) # B
    arr = convert_to_uint8(arr)
    return lut[arr]

def imshow(img: np.ndarray | Image.Image):
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = convert_to_uint8(img)
        img = Image.fromarray(img)
    img.show()

def export_tiff(img: np.ndarray | Image.Image, img_path: str, tiff_info: None):
    if isinstance(img, np.ndarray):
        if img.dtype is not np.uint8:
            img = convert_to_uint8(img)
        if img.ndim == 3:
            img = Image.fromarray(img, mode="RGB")
        else:
            img = Image.fromarray(img)
    if not isinstance(tiff_info, TiffImagePlugin.ImageFileDirectory_v2):
        tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
    path = Path(img_path).resolve()
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    basename = path.name.replace(path.suffix, ".tif")
    tiff_path = os.path.join(str(path.parent), basename)
    tiff_path = Path(tiff_path).as_posix()
    print(f"Export TIFF to: {tiff_path}")
    if len(img.getbands()) < 3:
        tiff_info[262] = 1 # PhotometricInterpretation, 1 means zero is black, 255 is white
        tiff_info[277] = 1 # SamplesPerPixel, 3 for RGB, 4 for RGBA
    img.save(tiff_path, format="tiff", tiffinfo=tiff_info)

def extract_root_region(arr: np.ndarray, kernel_size: int = 3):
    # Background must be black, target must be white
    kernel = sk.morphology.disk(kernel_size)
    mask = convert_to_uint8(arr)
    mask = sk_filters.rank.median(mask, kernel)
    mask = mask > sk_filters.threshold_otsu(mask)
    root_arr = np.multiply(arr, mask)
    return root_arr, mask