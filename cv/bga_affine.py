import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from typing import Union


def center_affine_transform(image_path: Union[str, np.ndarray], angle: float = 0, scale: float = 1.0):
    """
    brief:  Perform affine transformation on the image with the image center as the origin.
    param:  @image_path: Should be path or mat of image.
            @angle: The angle in degree at which the image needs to be rotated in a clockwise direction.
            @scale: Image scaling ratio.
    return: 1.Original image.
            2.Transformed image.
            3.Affine transformation mat.
    """
    if type(image_path) is str:
        if not os.path.exists(image_path):
            return None, None, None
    
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #threshold, processed img      img, thresh, maxval, threshold type 
        thresh, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    elif type(image_path) is np.ndarray:
        img = image_path
    else:
        return None, None, None
    height, width = img.shape[:2]
    
    # 将角度转换为弧度
    angle_rad = math.radians(angle)
    
    """
    [a00 a01 tx]
    [a10 a11 ty]
    """
    a00 = scale * math.cos(angle_rad)
    a01 = -scale * math.sin(angle_rad)
    a10 = scale * math.sin(angle_rad)
    a11 = scale * math.cos(angle_rad)
    
    # 计算旋转后的图像边界
    cos_val = abs(math.cos(angle_rad))
    sin_val = abs(math.sin(angle_rad))
    
    #新画布宽度和高度
    new_width = int((width * cos_val + height * sin_val) * scale)
    new_height = int((width * sin_val + height * cos_val) * scale)
    
    """
    新画布左上角还是原点,画布中心为(new_width/2, new_height/2)
    新图像的中心为:A*(width/2, height/2)^T
    水平方向移动单位:new_width/2 - (a00, a01)*(width/2, height/2)^T
    垂直方向移动单位:new_height/2 - (a10, a11)*(width/2, height/2)^T
    """
    tx = (new_width - width * a00 - height * a01) / 2
    ty = (new_height - width * a10 - height * a11) / 2
    
    # 构建完整的变换矩阵
    M = np.float32([[a00, a01, tx],
                    [a10, a11, ty]])
    
    # 应用变换到新的大小
    transformed_img = cv2.warpAffine(img, M, (new_width, new_height))
    
    return img, transformed_img, M
