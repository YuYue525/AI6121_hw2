import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

result_dir = "./results"

def BM(imgL, imgR, result_name = "default"):

    stereo = cv2.StereoBM_create(
        numDisparities = 16,
        blockSize = 15
    )
    disparity = stereo.compute(imgL, imgR)

    plt.imshow(disparity,'gray')
    plt.savefig(os.path.join(result_dir, result_name))

if __name__ == '__main__':

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    imgL_path = "triclopsi2l.jpg"
    imgR_path = "triclopsi2r.jpg"
    result_name = "BM_triclopsi2.png"
    
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)
    
    BM(imgL, imgR, result_name)
    
    imgL_path = "corridorl.jpg"
    imgR_path = "corridorr.jpg"
    result_name = "BM_corridor.png"
    
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)
    
    BM(imgL, imgR, result_name)
