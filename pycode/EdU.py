from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import *
from read_image import *

DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
use_cores = max(1, mp.cpu_count() - 2)

def EdU(czi_path: str, input_folder_path:str | None = None):
    img = ReadImage(czi_path)
    height = img.sizes.get("Y")
    width = img.sizes.get("X")
    mpp = round(img.mpp, 2) # micrometer per pixel
    EdU, TPMT = img.get_BES_arr() # The BES and EdU procedure is similar

    if input_folder_path is None:
        input_folder_path = Path(czi_path).resolve().parent.as_posix()
    else:
        input_folder_path = Path(input_folder_path).resolve().as_posix()

    output_folder_path = create_output_folder(input_folder_path, True)

    EdU_folder_path = os.path.join(output_folder_path, "EdU")
    EdU_img_path = czi_path.replace(input_folder_path, EdU_folder_path)
    zero_arr = np.zeros((height, width), dtype=np.uint8)
    EdU_RGB = np.stack([zero_arr, EdU, zero_arr], axis=2)
    export_tiff(EdU_RGB, EdU_img_path, img.tiff_info)
        
    if TPMT is not None:
        TPMT_folder_path = os.path.join(output_folder_path, "T-PMT")
        TPMT_img_path = czi_path.replace(input_folder_path, TPMT_folder_path)
        export_tiff(TPMT, TPMT_img_path, img.tiff_info)
        TPMT = np.bitwise_invert(TPMT)
        TPMT, mask = extract_root_region(TPMT)
        EdU = np.multiply(EdU, mask)

    if EdU.max() == 0:
        edu_total = 0
        edu_area = 0
        edu_mean = 0
    else:
        edu_total = EdU.sum()
        edu_area = np.sum(EdU > 0)
        edu_mean = round(edu_total / edu_area, 4)

    return {
        "img_dirname": img.img_path.parent.as_posix(),
        "img_basename": img.img_path.name,
        "channels": img.sizes.get("C"),
        "Zstacks": img.sizes.get("Z"),
        "img_size_pixels": f"{height} x {width}",
        "img_size_um": f"{height * mpp} x {width * mpp}",
        "resolution (um/pixel)": mpp,
        "BES_area": edu_area,
        "BES_mean": edu_mean,
        "BES_total": edu_total,
        "distance_pixels": -999,
        "note": "ok"
    }


def EdU_multproc(input_folder_path: str, use_cores: int = 3):
    input_folder_path = Path(input_folder_path).resolve().as_posix()
    img_list = get_img_list(input_folder_path, suffix=".czi")
    img_num = len(img_list)
    use_cores = min(img_num, use_cores)
    if img_num < 10 or use_cores < 3:
        csv_output = []
        for img in img_list:
            csv_output.append(EdU(img, input_folder_path))
    else:
        tasks = [(img, input_folder_path) for img in img_list]
        pool = mp.Pool(use_cores)
        csv_output = pool.starmap(EdU, tasks)
        pool.close()
        pool.join()
    
    output_folder = create_output_folder(input_folder_path, mkdir=False)
    csv_output_path = Path(output_folder) / f"OUT_EdU_{DATE_TIME}.csv"
    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv(csv_output_path, index=False)


# if __name__ == "__main__":
#     input_folder_path = "../test/EdU"

#     input_folder_path = Path(input_folder_path).resolve().as_posix()
#     img_list = get_img_list(input_folder_path)[1:3]

#     out = EdU_multproc(input_folder_path, use_cores)
