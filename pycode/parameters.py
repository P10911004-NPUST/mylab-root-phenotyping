# _dim_names: dict[str, str] = {
#     'T': 'time',
#     'Z': 'Z-stack',
#     'C': 'channel',
#     'H': 'phase',
#     'R': 'rotation',
#     'I': 'illumination',
#     'B': 'block',
#     'M': 'mosaic tile',
#     'A': 'acquisition',
#     'V': 'view',
# }

DEFAULT_DIMS = ["H", "T", "C", "Z", "Y", "X", "S"]

IMG_TYPE = {
    ".czi": "czi", 
    ".tif": "tiff", 
    ".tiff": "tiff", 
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".bmp": "bmp"
}

def NBT_output_dict():
    out = {
        "img_dirname": None,
        "img_basename": None,
        "width (pixels)": -999,
        "height (pixels)": -999,
        "um/pixel": -999.0,
        "nbt_area": -999,
        "nbt_mean": -999.0,
        "nbt_total": -999,
        "note": "ok"
    }
    return out
