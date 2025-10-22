import numpy as np
import cv2

"""
=================================
Zhang-Suen's thinning algorithm
"""
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    """
    the Zhang-Suen Thinning Algorithm
    输入图像需要为二值图像, 前景像素值为1, 背景像素值为0
    """
    Image_Thinned = image.copy()    # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if Image_Thinned[x][y] == 1 and sum(n) == 0:
                    changing1.append((x,y))
                    continue
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
"""
====================================
"""

def enhanced_upscale(image, target_size=(100, 100)):
    """
    增强型放大方法：预处理 + 高质量插值 + 后处理
    """
    # 步骤1: 预处理 - 锐化增强边缘
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # 步骤2: 使用高质量插值方法分阶段放大
    # 第一阶段：放大到中等尺寸
    intermediate_size = (target_size[0]//2, target_size[1]//2)
    stage1 = cv2.resize(sharpened, intermediate_size, interpolation=cv2.INTER_CUBIC)
    
    # 第二阶段：最终放大
    stage2 = cv2.resize(stage1, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 步骤3: 后处理 - 轻微锐化
    final = cv2.GaussianBlur(stage2, (0, 0), 1.0)           #高斯模糊处理
    final = cv2.addWeighted(stage2, 1.5, final, -0.5, 0)    #模糊图像和放大的图像加权叠加
    
    return final

def multi_scale_upscale(image, target_size=(100, 100)):
    """
    多尺度融合放大
    """
    # 生成多个尺度的放大结果
    scales = [2, 3, 4, 5]
    upscaled_images = []
    
    for scale in scales:
        # 计算当前尺度目标尺寸
        current_size = (target_size[0]//scale, target_size[1]//scale)       
        upscaled = cv2.resize(image, current_size, interpolation=cv2.INTER_CUBIC)
        # 再放大到目标尺寸
        final_upscaled = cv2.resize(upscaled, target_size, interpolation=cv2.INTER_LANCZOS4)
        upscaled_images.append(final_upscaled)
    
    # 融合多个结果（简单平均）
    fused = np.mean(upscaled_images, axis=0).astype(np.uint8)
    
    return fused

def denoising(image):
    height, width = image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (height // 20, width // 20))     #矩形卷积核
    denoised_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)                      #开运算去噪
    return denoised_img

def thickening(thin_img, ksize=(5, 5), iterations=2, preserve_features=True):
    """
    对细化后的数字进行加粗
    iterations越大、ksize越大, 线越粗
    """
    if preserve_features:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)    # 使用椭圆核，大小5*5。 更好地保持圆形特征
    else:
        kernel = np.ones(ksize, np.uint8)                               # 使用矩形核，更均匀的加粗
    
    stage1 = cv2.dilate(thin_img, kernel, iterations=iterations)        # 分阶段加粗
    
    # 轻微腐蚀以平滑边缘
    kernel_small = np.ones(tuple(map(lambda x: x // 2, ksize)), np.uint8)
    smoothed = cv2.morphologyEx(stage1, cv2.MORPH_OPEN, kernel_small)
    
    return smoothed

def digit_sharpening(image: str):
    """
    返回图像为0, 1化的矩阵.前景为文字像素, 值为1
    """
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = enhanced_upscale(img, (100, 100))                     #放大模糊图像,尺寸太大会增加细化时间
    th, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)   #二值化
    img = img < th                                              #01化并反转前景和背景
    img = denoising(img.astype(np.uint8))                       #去噪
    img = zhangSuen(img)                                        #细化
    img = img.astype(np.uint8)                                  #转换像素类型
    img = multi_scale_upscale(img, (500, 500))                  #第二次放大
    img = thickening(img)                                       #加粗前景
    return img                                                  #反转前景和背景

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = digit_sharpening('./images/dtcdig/1/1.png_cropped_2.jpg')
    print(img)
    plt.imshow(img)
    plt.show()