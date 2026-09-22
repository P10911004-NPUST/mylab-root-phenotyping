from warnings import WarningMessage
from datetime import datetime
import numpy as np
import pandas as pd
import skimage as sk
from PIL import Image, ImageDraw, ImageFont
# Explicitly import the module for pyinstaller
# Don't know why pyinstaller failed to bundle them otherwise
from skimage import filters as sk_filters
from skimage import transform as sk_transform
import multiprocessing as mp

from parameters import *
from utils import *
from read_image import *

DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
# use_cores = max(1, mp.cpu_count() - 2)

def estimate_kernel_size(arr: np.ndarray):
    kernel_size = np.sum(arr > 0) / (arr.shape[0] * arr.shape[1]) * 100 * 0.5
    kernel_size = np.ceil(kernel_size)
    # coerce to odd value
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    # kernel size at least 3
    kernel_size = max(3, kernel_size)
    return kernel_size

def NBT(img_path: str, input_folder_path: str | None = None, sensitivity: int = 2):
    out = NBT_output_dict()
    img_path = Path(img_path).resolve().as_posix()
    out["img_dirname"] = os.path.dirname(img_path)
    out["img_basename"] = os.path.basename(img_path)
    try:
        img = ReadImage(img_path)
        out["width (pixels)"] = img.sizes.get("X")
        out["height (pixels)"] = img.sizes.get("Y")
        out["um/pixel"] = round(img.mpp, 4) # micrometer per pixel
        RGB, GRAY = img.get_NBT_arr()
        
        if input_folder_path is None:
            input_folder_path = Path(img_path).resolve().parent.as_posix()
        else:
            input_folder_path = Path(input_folder_path).resolve().as_posix()

        output_folder_path = create_output_folder(input_folder_path, True)

        RGB_folder_path = Path(output_folder_path).joinpath("RGB").as_posix()
        RGB_img_path = img_path.replace(input_folder_path, RGB_folder_path)
        RGB_img_path = Path(RGB_img_path).with_suffix(".tif").as_posix()
        export_tiff(RGB, RGB_img_path, img.tiff_info)

        GRAY_folder_path = Path(output_folder_path).joinpath("GRAY").as_posix()
        GRAY_img_path = img_path.replace(input_folder_path, GRAY_folder_path)
        GRAY_img_path = Path(GRAY_img_path).with_suffix(".tif").as_posix()
        export_tiff(GRAY, GRAY_img_path, img.tiff_info)

        if np.std(GRAY) == 0:
            out["nbt_area"] = 0
            out["nbt_mean"] = 0
            out["nbt_total"] = 0
            out["note"] = "failed"
            return out
        
        # Reduce image size to accelerate computation
        sizeX = img.sizes.get("X")
        sizeY = img.sizes.get("Y")
        if sizeX > 600 and sizeY > 600:
            resize_X = np.int64(np.ceil(sizeX / 4))
            resize_Y = np.int64(np.ceil(sizeY / 4))
            GRAY_small = Image.fromarray(GRAY).resize(size=(resize_X, resize_Y))
            GRAY_small = np.asarray(GRAY_small)
        else:
            GRAY_small = GRAY

        kernel_size = estimate_kernel_size(GRAY_small)
        kernel = sk.morphology.disk(kernel_size)

        # Extract root region, this is the first step roughly to extract the NBT stained area
        GRAY_small, _ = extract_root_region(GRAY_small, kernel_size)

        # Extract ROI (the NBT stained area) from the root region
        # The opening operation is erosion followed by dilation.
        # I expect this operation will reduce the abnormal shape of the ROI, which
        # is usually caused by excessive staining in the middle.
        # But the other problem is, if the NBT solution not fully penetrated into
        # the middle tissue, then this operation may yield even more abnormal shape.
        # So this part needs to be further tested and optimized.
        sensitivity = min(max(2, 5 - sensitivity), 5)
        threshold = sk.filters.threshold_multiotsu(GRAY_small, classes=sensitivity)
        ROI = GRAY_small >= min(threshold)
        
        for _ in range(3):
            ROI = sk.morphology.opening(ROI, kernel)

        # keep only the largest blob
        blobs = sk.measure.label(ROI)
        blobs = sk.measure.regionprops(blobs)
        if not blobs:
            raise ValueError("No spots detected")
        area = [i.area for i in blobs]
        area = max(area) - 1
        ROI = sk.morphology.remove_small_objects(ROI, max_size=area)

        # Slightly enlarge the ROI to cover the NBT stained area more completely
        kernel2 = sk.morphology.disk(np.ceil(kernel_size ** 0.5))
        ROI = sk.morphology.dilation(ROI, kernel2)  # This is boolean

        # Resize the ROI back to the original dimension
        #ROI = sk.transform.resize(ROI, (height, width), anti_aliasing=False, preserve_range=True)
        ROI = Image.fromarray(np.uint8(ROI)).resize((GRAY.shape[1], GRAY.shape[0]))
        ROI = np.asarray(ROI)
        NBT_GRAY = np.multiply(GRAY, ROI)

        nbt_area = ROI.sum(dtype=np.double)
        nbt_total = NBT_GRAY.sum(dtype=np.double)
        nbt_mean = nbt_total / nbt_area if nbt_area > 0 else 0
        out["nbt_area"] = nbt_area.astype(np.int64)
        out["nbt_total"] = nbt_total.astype(np.int64)
        out["nbt_mean"] = round(nbt_mean, 4)
        out["note"] = "failed" if nbt_area == 0 else "ok"

        NBT_RGB = Image.fromarray(RGB)
        
        draw = ImageDraw.Draw(NBT_RGB)
        contours = sk.measure.find_contours(ROI, level = 0)
        contour_color = (0, 166, 251) if RGB.ndim == 3 else 255
        nbt_mean = round(out["nbt_mean"], 2)
        nbt_total = round(out["nbt_total"], 2)
        nbt_area = round(out["nbt_area"], 2)
        text = f"Avg: {round(nbt_mean, 2)} = {round(nbt_total/1_000_000, 2)} M / {nbt_area} pixels"
        font_settings = ImageFont.load_default(size=90)
        if img.sizes.get("X") < 800 or img.sizes.get("Y") < 800:
            font_settings = ImageFont.load_default(size=30)
        for contour in contours:
            points = [(c[1], c[0]) for c in contour]    # (x, y)
            draw.line(points, fill = contour_color, width = 7)
            draw.text(xy = (30, 10), text = text, fill = contour_color, font = font_settings)

        NBT_folder_path = Path(output_folder_path).joinpath("NBT").as_posix()
        NBT_img_path = img_path.replace(input_folder_path, NBT_folder_path)
        NBT_img_path = Path(NBT_img_path).with_suffix(".jpg").as_posix()
        if not os.path.exists(os.path.dirname(NBT_img_path)):
            os.makedirs(os.path.dirname(NBT_img_path))
        NBT_RGB.save(NBT_img_path, format="JPEG")
    except:
        out["note"] = "failed"

    return out


def NBT_multproc(input_folder_path: str, use_cores: int = 3):
    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path, suffix=IMG_TYPE)
    img_num = len(img_list)
    use_cores = min(img_num, use_cores)
    if img_num < 5 or use_cores < 3:
        csv_output = []
        for img in img_list:
            # the output will be a list of dictionaries
            csv_output.append(NBT(img, input_folder_path))
    else:
        tasks = [(img, input_folder_path) for img in img_list]
        pool = mp.Pool(use_cores)
        # the output is a list of dictionaries
        csv_output = pool.starmap(NBT, tasks)
        pool.close()
        pool.join()

    output_folder = create_output_folder(input_folder_path, mkdir=False)
    csv_output_path = Path(output_folder) / f"OUT_NBT_{DATE_TIME}.csv"
    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv(csv_output_path, index=False)


# czi_path = "../test/NBT/ath_nbt-reddish.czi"
# tiff_path = "../test/NBT/ath_nbt_01.tif"
# png_path = "../test/NBT/rice_nbt_gray.png"

if __name__ == "__main__":
    input_folder_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing"

    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path)

    out = NBT_multproc(input_folder_path, 15)

