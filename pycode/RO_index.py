import numpy as np
import skimage as sk
# Explicitly import the module for pyinstaller
# Don't know why pyinstaller failed to bundle them otherwise
from skimage import filters as sk_filters
from skimage import transform as sk_transform

from read_image import *
from utils import *

# def extract_root_region(arr: np.ndarray, kernel_size: int = 3):
#     # Background must be black, target must be white
#     kernel = sk.morphology.disk(kernel_size)
#     mask = convert_to_uint8(arr)
#     mask = sk_filters.rank.median(mask, kernel)
#     mask = mask > sk_filters.threshold_otsu(mask)
#     root_arr = np.multiply(arr, mask)
#     return root_arr, mask

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
    E405, E488, PI = img.get_RO_related_arr()
    RO = calc_RO_index(E405, E488)
    LUT = gray_to_lut(RO)
    
    if input_folder_path is None:
        # input_folder_path = os.path.dirname(czi_path)
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
