import cv2
from glob import glob
import os

from config import Config

card_size = (63, 88)

#TODO: Figure out why this is here
for subdir in glob(Config.data_dir + "/card_img/png/*"):
    split = subdir.split('/')
    split[-2] = 'tiny'
    dir_out = '/'.join(split)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for f_in in glob(subdir + "/*.png"):
        print(f_in)
        f_out = dir_out + '/' + os.path.split(f_in)[1]
        if not os.path.exists(f_out):
            png = cv2.imread(f_in)
            png_resize = cv2.resize(png, card_size)
            cv2.imwrite(f_out, png_resize)
    print(subdir)
