from utils import convert_to_uint8
from utils import *

class ReadImage:
    def __init__(self, img_path: str):
        self.tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
        self.img_path = Path(img_path).resolve()
        self.mpp = -999
        self._load_img()

    @property
    def img_type(self) -> str:
        return IMG_TYPE.get(self.img_path.suffix.lower(), "unknown")

    def _load_img(self) -> None:
        if self.img_type == "czi":
            self.read_czi()
        elif self.img_type in ["tif", "tiff"]:
            self.read_tiff()
        else:
            self.read_other_image_type()

    def read_czi(self):
        if self.img_type != "czi":
            raise ValueError(f"File '{self.img_path}' is not a CZI file.")
        czi = czifile.CziFile(str(self.img_path), squeeze=False)
        self.scenes = czi.scenes()
        self.arr = self.scenes.asarray()
        self.dims = self.scenes.dims
        self.sizes = self.scenes.sizes
        # ZEISS_AxioScopeA1: (C, Y, X, S)
        self.machine = "ZEISS_AxioScopeA1" if len(self.dims) <= 4 else "confocal"
        self.tiff_info[256] = self.sizes.get("X") # ImageWidth
        self.tiff_info[257] = self.sizes.get("Y") # ImageLength
        self.tiff_info[282] = 10_000 / self.scenes.mpp[0] # XResolution (pixels per cm)
        self.tiff_info[283] = 10_000 / self.scenes.mpp[1] # YResolution (pixels per cm)
        self.tiff_info[296] = 3  # ResolutionUnit: 1 = none, 2 = inch, 3 = centimeter
        self.mpp = self.scenes.mpp[0]
        czi.close()

    def read_tiff(self):
        if self.img_type not in ["tif", "tiff"]:
            raise ValueError(f"File '{self.img_path}' is not a TIFF file.")
        self.scenes = Image.open(str(self.img_path))  # (height, width, channel)
        self.arr = np.asarray(self.scenes)
        if self.arr.ndim == 2:
            self.arr = np.expand_dims(self.arr, 2)
        self.arr = np.transpose(self.arr, (2, 0, 1))
        self.dims = ("C", "Y", "X")
        self.width = self.scenes.width
        self.height = self.scenes.height
        self.channel = len(self.scenes.getbands())
        self.sizes = {"C": self.channel, "Y": self.height, "X": self.width}
        self.machine = "unknown"
        self.tiff_info = self.scenes.tag_v2
        resolution_unit = self.tiff_info[296] if self.tiff_info.get(296) is not None else 1
        if resolution_unit == 1:
            self.mpp = np.double(1 / self.tiff_info[282])
        if resolution_unit == 3:
            self.mpp = np.double(10_000 / self.tiff_info[282])
    
    def read_other_image_type(self):
        self.scenes = Image.open(str(self.img_path)) # default is (height, width, channel)
        self.arr = np.asarray(self.scenes)
        if self.arr.ndim == 2:
            self.arr = np.expand_dims(self.arr, 2)
        self.arr = np.transpose(self.arr, (2, 0, 1)) # transpose to (channel, height, width)
        self.dims = ("C", "Y", "X")
        self.width = self.scenes.width
        self.height = self.scenes.height
        self.channel = len(self.scenes.getbands())
        self.sizes = {"C": self.channel, "Y": self.height, "X": self.width}
        self.machine = "unknown"
        self.tiff_info[256] = self.width # ImageWidth
        self.tiff_info[257] = self.height # ImageLength
    
    def check_img_dims(self, default_dims = ("H", "T", "C", "Z", "Y", "X", "S")):
        exists_dims = {}
        for key in default_dims:
            if key in self.dims and self.sizes.get(key) > 1:
                exists_dims[key] = True
            else:
                exists_dims[key] = False
        return exists_dims

    def get_NBT_arr(self):
        if self.machine == "ZEISS_AxioScopeA1":
            # (C, Y, X, S) --> 'S' is the RGB channel
            # RGB = np.mean(self.arr, 0)
            # GRAY = convert_to_uint8(np.mean(RGB, -1))
            R = self.arr[0, :, :, 0].astype(np.double)
            G = self.arr[0, :, :, 1].astype(np.double)
            B = self.arr[0, :, :, 2].astype(np.double)
            RGB = convert_to_uint8(np.stack([R, G, B], axis=2))
            GRAY = convert_to_uint8((B + B + R + G) / 4)
            # RGB = np.stack([R, G, B], axis=2)
            # GRAY = np.mean(self.arr, 3).squeeze()
            # GRAY = convert_to_uint8(GRAY)
        if self.machine == "confocal":
            gray = self.arr
            if self.sizes.get("C") > 1:
                idx = list(self.sizes).index("C")
                # gray = np.mean(self.arr, 2, keepdims=True)
                gray = np.mean(self.arr, 2, keepdims=True)
            if self.sizes.get("S") > 1:
                idx = list(self.sizes).index("S")
                # gray = np.mean(self.arr, 6, keepdims=True)
                gray = np.mean(self.arr, idx, keepdims=True)
            GRAY = gray.squeeze()
            GRAY = convert_to_uint8(GRAY)
            RGB = np.stack([GRAY, GRAY, GRAY], axis=-1)
        if self.machine == "unknown":
            RGB = self.arr.squeeze()
            if RGB.ndim == 2:
                GRAY = convert_to_uint8(RGB)
                RGB = np.stack([GRAY, GRAY, GRAY], axis=2)
            elif RGB.ndim == 3:
                R = self.arr[0, :, :].astype(np.double)
                G = self.arr[1, :, :].astype(np.double)
                B = self.arr[2, :, :].astype(np.double)
                RGB = np.stack([R, G, B], axis=2)
                GRAY = convert_to_uint8((B + B + R + G) / 4)
                # GRAY = convert_to_uint8(np.mean(RGB, 0).squeeze())
                # RGB = np.transpose(RGB, (1, 2, 0))
            else:
                return self.arr
        # The exported array dimensions are (height, width, channel)
        return convert_to_uint8(RGB), np.bitwise_invert(GRAY)

    def get_EdU_arr(self):
        if "Z" not in self.dims or self.machine != "confocal":
            raise ValueError("Z-stack is required for the EdU protocol.")
        GRAY = np.max(self.arr, axis=self.dims.index("Z")).squeeze()
        GRAY = convert_to_uint8(GRAY)
        # The exported array dimensions are (height, width)
        return GRAY

    def get_BES_arr(self, BES_idx = 0, TPMT_idx = 1):
        arr = self.arr
        if self.dims.index("Z") is not None:
            arr = np.max(arr, axis=self.dims.index("Z"), keepdims=True)
        BES = convert_to_uint8(arr[0, 0, BES_idx, 0, :, :, 0])
        TPMT = None
        if self.sizes.get("C") > 1:
            TPMT = convert_to_uint8(arr[0, 0, TPMT_idx, 0, :, :, 0])
        # BES = self.arr[:, :, 0:1, :, :, :, :]
        # arr = np.max(arr, axis=self.dims.index("Z"))
        return BES, TPMT

    def get_RO_related_arr(self, E405_idx = 0, E488_idx = 1, PI_idx = 2):
        arr = self.arr
        if self.sizes.get("Z") > 1:
            arr = np.max(arr, axis=self.dims.index("Z"), keepdims=True)
        if self.sizes.get("S") > 1:
            arr = np.max(arr, axis=self.dims.index("S"), keepdims=True)
        E405 = arr[0, 0, E405_idx, 0, :, :, 0]
        E488 = arr[0, 0, E488_idx, 0, :, :, 0]
        PI = None
        if self.sizes.get("C") > 2:
            PI = convert_to_uint8(arr[0, 0, PI_idx, 0, :, :, 0])
        return E405, E488, PI
