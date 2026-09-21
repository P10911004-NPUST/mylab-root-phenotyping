from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import *
from read_image import *

DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
use_cores = max(1, mp.cpu_count() - 2)

def percentile_filter(arr: np.ndarray, perc: tuple = (0, 99)):
    # Discard saturated pixels, especially the root cap
    raw_arr = arr
    arr = np.double(arr)
    bg_val = arr.min()
    lower_threshold = np.percentile(arr, perc[0])
    upper_threshold = np.percentile(arr, perc[1])
    ratio_of_unique_val = len(np.unique(arr)) / len(arr)
    if ratio_of_unique_val > 0.3:
        arr[raw_arr < lower_threshold] = bg_val
        arr[raw_arr > upper_threshold] = bg_val
    return arr


def calc_RO_index(E405, E488):
    e405, e405_mask = extract_root_region(E405)
    e488, e488_mask = extract_root_region(E488)
    protein_exists_mask = np.bitwise_or(e405_mask, e488_mask)
    e405 = percentile_filter(e405) + 1
    e488 = percentile_filter(e488) + 1
    e488 = e488 * np.max(e405) / np.max(e488)
    # e488 = np.nan_to_num(e488, nan=0)
    e405 = e405 ** 2
    e488 = e488 ** 2
    RO = np.divide(e405 - e488, e405 + e488)
    # numer = e405 - e488
    # denom = e405 + e488
    # # suppress warnings when divided by zero
    # RO = np.zeros_like(e405, dtype=np.double)
    # RO = np.divide(numer, denom, out = RO, where=denom != 0)
    RO = RO * protein_exists_mask
    return RO


def RO_index(czi_path: str, input_folder_path: str | None = None):
    img = ReadImage(czi_path)
    try:
        height = img.sizes.get("Y")
        width = img.sizes.get("X")
        mpp = round(img.mpp, 2) # micrometer per pixel
        E405, E488, PI = img.get_RO_related_arr()
        RO = calc_RO_index(E405, E488)
        LUT = gray_to_lut(RO)
        
        if input_folder_path is None:
            input_folder_path = Path(czi_path).resolve().parent.as_posix()
        else:
            input_folder_path = Path(input_folder_path).resolve().as_posix()

        output_folder_path = create_output_folder(input_folder_path, True)
        # E405: Oxidized-state protein signal
        E405_folder_path = os.path.join(output_folder_path, "405 nm")
        E405_img_path = czi_path.replace(input_folder_path, E405_folder_path)
        export_tiff(E405, E405_img_path, img.tiff_info)
        # 488 nm -> Reduced-state protein signal
        E488_folder_path = os.path.join(output_folder_path, "488 nm")
        E488_img_path = czi_path.replace(input_folder_path, E488_folder_path)
        export_tiff(E488, E488_img_path, img.tiff_info)
        # Propidium Iodide stain
        if PI is not None:
            PI_folder_path = os.path.join(output_folder_path, "PI")
            PI_img_path = czi_path.replace(input_folder_path, PI_folder_path)
            export_tiff(PI, PI_img_path, img.tiff_info)
        # Reduced-Oxidized index
        RO_folder_path = os.path.join(output_folder_path, "RO")
        RO_img_path = czi_path.replace(input_folder_path, RO_folder_path)
        export_tiff(RO, RO_img_path, img.tiff_info)
        # Look-up-table pseudo-image
        LUT_folder_path = os.path.join(output_folder_path, "LUT")
        LUT_img_path = czi_path.replace(input_folder_path, LUT_folder_path)
        export_tiff(LUT, LUT_img_path, img.tiff_info)

        return {
            "img_dirname": img.img_path.parent.as_posix(),
            "img_basename": img.img_path.name,
            "channels": img.sizes.get("C"),
            "Zstacks": img.sizes.get("Z"),
            "img_size_pixels": f"{height} x {width}",
            "img_size_um": f"{height * mpp} x {width * mpp}",
            "resolution (um/pixel)": mpp,
            "note": "ok"
        }
    except ValueError as err:
        print(err)
        return {
            "img_dirname": img.img_path.parent.as_posix(),
            "img_basename": img.img_path.name,
            "channels": -999,
            "Zstacks": -999,
            "img_size_pixels": -999,
            "img_size_um": -999,
            "resolution (um/pixel)": -999,
            "note": "failed"
        }


def RO_index_multproc(input_folder_path: str, use_cores: int = 3):
    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path, suffix=".czi")
    img_num = len(img_list)
    use_cores = min(img_num, use_cores)
    if img_num < 10 or use_cores < 3:
        csv_output = []
        for img in img_list:
            # the output will be a list of dictionaries
            csv_output.append(RO_index(img, input_folder_path))
    else:
        tasks = [(img, input_folder_path) for img in img_list]
        pool = mp.Pool(use_cores)
        # the output is a list of dictionaries
        csv_output = pool.starmap(RO_index, tasks)
        pool.close()
        pool.join()

    output_folder = create_output_folder(input_folder_path, mkdir=False)
    csv_output_path = Path(output_folder) / f"OUT_RO-index_{DATE_TIME}.csv"
    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv(csv_output_path, index=False)

# if __name__ == "__main__":
#     input_folder_path = "../test/RO-index"

#     input_folder_path = Path(input_folder_path).resolve().as_posix()
#     img_list = get_img_list(input_folder_path)

#     out = RO_index_multproc(input_folder_path, use_cores)