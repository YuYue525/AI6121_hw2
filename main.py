import os
import math
import numpy as np
from PIL import Image

def sum_of_squares_difference(pos_x, pos_y, img_1, img_2, patch_size = (3, 3)):
    # img_1, img_2 are Image objects have the same img size
    img_width, img_height = img_1.size
    
    img_1_array = np.array(img_1)
    img_2_array = np.array(img_2)
    
    line_ssd = []
    # for the patch on the line
    for i in range(img_width):
        patch_ssd = 0
        
        for patch_h in range(patch_size[0]):
            for patch_w in range(patch_size[1]):
                # positions of the original patch pixel and the target patch pixel
                origin_pos = (int(pos_x - (patch_size[0] - 1)/2 + patch_h), int(pos_y - (patch_size[1] - 1)/2 + patch_w))
                target_pos = (int(pos_x - (patch_size[0] - 1)/2 + patch_h), int(i - (patch_size[1] - 1)/2 + patch_w))
                # if pixels are in the images
                if origin_pos[0] >= 0 and origin_pos[0] < img_height and origin_pos[1] >= 0 and origin_pos[1] < img_width and target_pos[0] >= 0 and target_pos[0] < img_height and target_pos[1] >= 0 and target_pos[1] < img_width :
                    patch_ssd += (int(img_1_array[origin_pos[0]][origin_pos[1]][0]) - int(img_2_array[target_pos[0]][target_pos[1]][0])) ** 2
        
        line_ssd.append(patch_ssd)
    
    threshold_dis = 20
        
    line_ssd = [line_ssd[x] + (abs(x - pos_y) > threshold_dis) * 100000000 for x in range(len(line_ssd))]
    # print(line_ssd)
        
    return abs(pos_y - np.argmin(line_ssd))
    
def draw_disparity_map(img_1, img_2, patch_size = (3, 3)):
    img_width, img_height = img_1.size
    map = np.zeros([img_height, img_width, 3])
    '''
    for i in range(img_height // patch_size[0]):
        for j in range(img_width // patch_size[1]):
            pixel_disparity = sum_of_squares_difference(patch_size[0] * i + (patch_size[0] - 1)/2, patch_size[1] * j + (patch_size[1] - 1)/2, img_1, img_2, patch_size)
            # print(pixel_disparity)
            map[(patch_size[0]*i):(patch_size[0]*(i+1)), (patch_size[1]*j):(patch_size[1]* (j+1)), :] = (pixel_disparity - 10) / 10 * 64 + 64
    '''
    
    for i in range(img_height):
        for j in range(img_width):
            pixel_disparity = sum_of_squares_difference(i, j, img_1, img_2, patch_size)
            # print(pixel_disparity)
            map[i, j, :] = (pixel_disparity - 10) / 10 * 64 + 64
        
    disparity_image = Image.fromarray(np.uint8(map))
    disparity_image.save("result.png")


if __name__ == '__main__':
    img_path_1 = "triclopsi2l.jpg"
    img_1 = Image.open(img_path_1)
    img_path_2 = "triclopsi2r.jpg"
    img_2 = Image.open(img_path_2)

    draw_disparity_map(img_1, img_2)
