from datetime import datetime
import csv
import numpy as np
import skimage as sk
# Explicitly import the module for pyinstaller
# Don't know why pyinstaller failed to bundle them otherwise
from skimage import filters as sk_filters
from skimage import transform as sk_transform
import multiprocessing as mp

from read_image import *
from utils import *

DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

BES_DATA = {
    "img_dirname": [],
    "img_basename": [],
    "channels": [],
    "Zstacks": [],
    "img_size_pixels": [],
    "img_size_um": [],
    "resolution (um/pixel)": [],
    "BES_area": [],
    "BES_mean": [],
    "BES_total": [],
    "distance_pixels": [],
    "note": []
}

input_folder = "../test/BES-H2O2-Ac/"
input_file = "../test/BES-H2O2-Ac/BES.czi"

img_list = get_img_list(input_folder)

def BES_H2O2_Ac(czi_path: str, input_folder_path:str | None = None):
    img = ReadImage(czi_path)
    BES, TPMT = img.get_BES_arr()
    height = img.sizes.get("Y")
    width = img.sizes.get("X")
    mpp = round(img.scenes.mpp[0], 2) # micrometer per pixel

    if input_folder_path is None:
        input_folder_path = Path(czi_path).resolve().parent.as_posix()
    else:
        input_folder_path = Path(input_folder_path).resolve().as_posix()

    output_folder_path = create_output_folder(input_folder_path, True)

    BES_folder_path = os.path.join(output_folder_path, "BES")
    BES_img_path = czi_path.replace(input_folder_path, BES_folder_path)
    print(f"Export TIFF: {BES_img_path}")
    export_tiff(BES, BES_img_path, img.tiff_info)
        
    if TPMT is not None:
        TPMT_folder_path = os.path.join(output_folder_path, "T-PMT")
        TPMT_img_path = czi_path.replace(input_folder_path, TPMT_folder_path)
        print(f"Export TIFF: {TPMT_img_path}")
        export_tiff(TPMT, TPMT_img_path, img.tiff_info)
        TPMT = np.bitwise_invert(TPMT)
        TPMT, mask = extract_root_region(TPMT)
        BES = np.multiply(BES, mask)

    if BES.max() == 0:
        bes_total = 0
        bes_area = 0
        bes_mean = 0
    else:
        bes_total = BES.sum()
        bes_area = np.sum(BES > 0)
        bes_mean = round(bes_total / bes_area, 4)

    BES_DATA["img_dirname"].append(img.img_path.parent.as_posix())
    BES_DATA["img_basename"].append(img.img_path.name)
    BES_DATA["channels"].append(img.sizes.get("C"))
    BES_DATA["Zstacks"].append(img.sizes.get("Z"))
    BES_DATA["img_size_pixels"].append(f"{height} x {width}")
    BES_DATA["img_size_um"].append(f"{height * mpp} x {width * mpp}")
    BES_DATA["resolution (um/pixel)"].append(mpp)
    BES_DATA["BES_area"].append(bes_area)
    BES_DATA["BES_mean"].append(bes_mean)
    BES_DATA["BES_total"].append(bes_total)
    BES_DATA["distance_pixels"].append(0)
    BES_DATA["note"].append("ok")

def BES_H2O2_Ac_multproc(input_folder_path: str, use_cores: int = 3):
    img_list = get_img_list(input_folder_path, suffix=".czi")
    print(img_list)

BES_H2O2_Ac_multproc(input_folder)
