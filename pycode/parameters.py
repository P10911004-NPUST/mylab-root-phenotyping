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

