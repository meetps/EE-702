import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import  *

HOME_PATH = "/home/meetshah1995/Desktop/Dropbox/Third_Year/II/EE702-Computer_Vision/EE-702/stereo-vision/data/"
# data_dirs = ['Baby1', 'Baby2', 'Baby3', 'Bowling1', 'Bowling2', 'Cloth1', 'Cloth2', 'Cloth3', 'Cloth4', 'Flowerpots', 'Lampshade1', 'Lampshade2', 'Midd1', 'Midd2', 'Monopoly', 'Plastic', 'Rocks1', 'Rocks2', 'Wood1', 'Wood2']
data_dirs = ['Wood2']

def stereoMatchSSD(left_img, right_img,directory):
    left_img = Image.open(left_img)
    left = np.asarray(left_img)
    right_img = Image.open(right_img)
    right = np.asarray(right_img)

    w, h = left_img.size
    sd = np.empty((w, h), np.uint32)
    sd.shape = h, w

    win_ssd = np.empty((w, h), np.uint32)
    win_ssd.shape = h, w
    
    depth = np.empty((w, h), np.uint32)
    depth.shape = h, w
    min_ssd = np.empty((w, h), np.uint32)
    min_ssd.shape = h, w
    for y in range(h):
        for x in range(w):
            min_ssd[y, x] = 65535 

    max_offset = 30
    offset_adjust = 255 / max_offset

    y_range = range(h)
    x_range = range(w)
    x_range_ssd = range(w)

    window_range = range(-3, 3)
    for offset in tqdm(range(max_offset)):
        for y in y_range:
            for x in x_range_ssd:
                if x - offset > 0:
                    diff = left[y, x,0] - right[y, x - offset,0]
                    sd[y, x] = diff * diff
        for y in y_range:
            for x in x_range:
                sum_sd = 0
                for i in window_range:
                    for j in window_range:
                        if (-1 < y + i < h) and (-1 < x + j < w):
                            sum_sd += sd[y + i, x + j]
                win_ssd[y, x] = sum_sd
        for y in y_range:
            for x in x_range:
                if win_ssd[y, x] < min_ssd[y, x]:
                    min_ssd[y, x] = win_ssd[y, x]
                    depth[y, x] = offset * offset_adjust

        # print("Calculated offset ", offset)
    imgplot = plt.imshow(depth,cmap="hot")
    plt.savefig(HOME_PATH + directory + '/depth_SSD.png')
    plt.show()

if __name__ == '__main__':
    for directory in data_dirs:
        imageleft = HOME_PATH + directory + '/view0.png'
        imageright = HOME_PATH + directory + '/view1.png'
        print imageright
        stereoMatchSSD(imageleft, imageright,directory)
