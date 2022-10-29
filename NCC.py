import os
import stereo
import scipy.misc
from PIL import Image
from pylab import *
from scipy.ndimage import *

result_dir = "./results"

def NCC(im_l, im_r, start, steps, wid):
    # calculating disparity images using Normalized cross-correlation.
    m, n = im_l.shape
    # arrays holding different summation values
    mean_l = zeros((m, n))
    mean_r = zeros((m, n))
    s = zeros((m, n))
    s_l = zeros((m, n))
    s_r = zeros((m, n))
    # holds an array of depth planes
    dmaps = zeros((m, n, steps))
    # calculate the average of the image blocks
    uniform_filter(im_l, wid, mean_l)
    uniform_filter(im_r, wid, mean_r)
    # normalize image
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # try different parallax
    for displ in range(steps):
        # move the left image to the right and calculate the sum
        uniform_filter(roll(norm_l, -displ - start) * norm_r, wid, s)  # and normalization
        uniform_filter(roll(norm_l, -displ - start) * roll(norm_l, -displ - start), wid, s_l)
        uniform_filter(norm_r * norm_r, wid, s_r)  # and inverse normalization
        # save the ncc score
        dmaps[:, :, displ] = s / sqrt(s_l * s_r)
    # select the best depth for each pixel
    return argmax(dmaps, axis=2)

if __name__ == '__main__':

    imgL_path = "triclopsi2l.jpg"
    imgR_path = "triclopsi2r.jpg"
    # imgL_path = "corridorl.jpg"
    # imgR_path = "corridorr.jpg"
    
    
    result_name = "triclopsi2"
    # result_name = "corridor"

    im_l = array(Image.open(imgL_path).convert('L'), 'f')
    im_r = array(Image.open(imgR_path).convert('L'), 'f')
    
    # start offset and set the step
    steps = 20
    start = 0

    # ncc width
    wid = 27

    res = NCC(im_l, im_r, start, steps, wid)

    imsave(os.path.join(result_dir, "NCC_" + result_name + "_patch_" + str(wid) + "_range_" + str(steps) + ".jpg"), res, cmap='gray')


