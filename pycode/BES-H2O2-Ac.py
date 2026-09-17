from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
import skimage as sk
from skimage import filters as sk_filters

from utils import *
from read_image import *

DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
use_cores = max(1, mp.cpu_count() - 2)

def BES_H2O2_Ac(czi_path: str, input_folder_path:str | None = None):
    czi_path = Path(czi_path).resolve().as_posix()
    img = ReadImage(czi_path)
    height = img.sizes.get("Y")
    width = img.sizes.get("X")
    mpp = round(img.scenes.mpp[0], 2) # micrometer per pixel
    BES, TPMT = img.get_BES_arr()

    if input_folder_path is None:
        input_folder_path = Path(czi_path).resolve().parent.as_posix()
    else:
        input_folder_path = Path(input_folder_path).resolve().as_posix()

    output_folder_path = create_output_folder(input_folder_path, True)

    BES_folder_path = os.path.join(output_folder_path, "BES")
    BES_img_path = czi_path.replace(input_folder_path, BES_folder_path)
    export_tiff(BES, BES_img_path, img.tiff_info)
        
    if TPMT is not None:
        TPMT_folder_path = os.path.join(output_folder_path, "T-PMT")
        TPMT_img_path = czi_path.replace(input_folder_path, TPMT_folder_path)
        export_tiff(TPMT, TPMT_img_path, img.tiff_info)
        TPMT = np.bitwise_invert(TPMT)
        TPMT, mask = extract_root_region(TPMT)
        BES = np.multiply(BES, mask)
        fg = BES > sk.filters.threshold_otsu(BES)
        bg = TPMT * (1.0 - fg)
        R = convert_to_uint8(bg)
        G = convert_to_uint8(bg + (fg * 255.0))
        B = convert_to_uint8(bg)
        RGB = np.stack([R, G, B], axis=2)
        pseudo_folder_path = os.path.join(output_folder_path, "BES_pseudo")
        pseudo_img_path = czi_path.replace(input_folder_path, pseudo_folder_path)
        export_tiff(RGB, pseudo_img_path, img.tiff_info)

    if BES.max() == 0:
        bes_total = 0
        bes_area = 0
        bes_mean = 0
    else:
        bes_total = BES.sum()
        bes_area = np.sum(BES > 0)
        bes_mean = round(bes_total / bes_area, 4)

    return {
        "img_dirname": img.img_path.parent.as_posix(),
        "img_basename": img.img_path.name,
        "channels": img.sizes.get("C"),
        "Zstacks": img.sizes.get("Z"),
        "img_size_pixels": f"{height} x {width}",
        "img_size_um": f"{height * mpp} x {width * mpp}",
        "resolution (um/pixel)": mpp,
        "BES_area": bes_area,
        "BES_mean": bes_mean,
        "BES_total": bes_total,
        "distance_pixels": -999,
        "note": "ok"
    }


def BES_H2O2_Ac_multproc(input_folder_path: str, use_cores: int = 3):
    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path, suffix=".czi")
    img_num = len(img_list)
    use_cores = min(img_num, use_cores)
    if img_num < 10 or use_cores < 3:
        csv_output = []
        for img in img_list:
            csv_output.append(BES_H2O2_Ac(img, input_folder_path))
    else:
        tasks = [(img, input_folder_path) for img in img_list]
        pool = mp.Pool(use_cores)
        csv_output = pool.starmap(BES_H2O2_Ac, tasks)
        pool.close()
        pool.join()
    
    output_folder = create_output_folder(input_folder_path, mkdir=False)
    csv_output_path = Path(output_folder) / f"OUT_BES-H2O2-Ac_{DATE_TIME}.csv"
    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv(csv_output_path, index=False)


if __name__ == "__main__":
    input_folder_path = "../test/BES-H2O2-Ac"

    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path)[1:3]

    out = BES_H2O2_Ac_multproc(input_folder_path, use_cores)
