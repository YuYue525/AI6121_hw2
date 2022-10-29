import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

result_dir = "./results"

if __name__ == '__main__':

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    '''
    imgL_path = "triclopsi2l.jpg"
    imgR_path = "triclopsi2r.jpg"
    result_name = "SGMB_triclopsi2.png"
    '''
    imgL_path = "corridorl.jpg"
    imgR_path = "corridorr.jpg"
    result_name = "SGMB_corridor.png"
    
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)

    # disparity range tuning
    window_size = 5
    min_disp = 0
    num_disp = 16 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize = 5,
        P1 = 8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2 = 32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    
    plt.imshow(disparity, 'gray')
    plt.savefig(os.path.join(result_dir, result_name))
