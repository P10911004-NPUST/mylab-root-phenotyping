from utils import get_img_list
from RO_index import *

input_folder = "../test/czi"
img_list = get_img_list(input_folder)

for img in img_list:
    RO_index(img, input_folder)


