import cv2
import numpy as np
import math
import os
from typing import Union
from blurred_digimg_processing import digit_sharpening

template_0 = np.int32([[[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1]],    #1    (50,50)
                        [[1, 0, 1], [1, 0, 2], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0]],   #2    (50,50)
                        [[1, 0, 1], [1, 1, 1], [1, 2, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1]],   #3    (50,50)
                        [[0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1]],   #4    (50,50)
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1]],   #5    (30,30)
                        [[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1]],   #6    (80,80)
                        [[1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]],   #7    (100,100)
                        [[1, 2, 1], [1, 1, 1], [1, 2, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1]],   #8    (50,50)
                        [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0]]])  #9    (50,50)

template_90 = np.int32([[[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1]],
                        [[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [2, 0, 1], [1, 1, 1]],
                        [[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 2, 1]],
                        [[1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]],
                        [[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]],
                        [[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 2]],
                        [[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 2, 1], [1, 1, 1], [1, 2, 1]],
                        [[0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]]])

template_180 = np.int32([[[1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
                        [[1, 2, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]],
                        [[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                        [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]],
                        [[0, 0, 2], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
                        [[1, 2, 1], [1, 1, 1], [1 ,2, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1]],
                        [[0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]]])
digit = 8
angle = 270
template_270 = np.int32([[[1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]],
                        [[1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 2], [1, 0, 1]],
                        [[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 2, 1], [1, 1, 1], [1, 0, 1]],
                        [[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1]],
                        [[0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]],
                        [[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 2, 1], [1, 1, 1], [1, 2, 1]],
                        [[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0]]])

templates = [template_0, template_90, template_180, template_270]

sub_template_0 = np.int32([1])

def crop_bounding_rect(image: cv2.typing.MatLike):
    #切取数字的最小外接矩形
    dummy, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)    #可能需要调整二值阈值
    img_inv = cv2.bitwise_not(image)                            #颜色反转,识别边框需要白色在内部
    contours, dummy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)       #获取轮廓
    largest_contour = max(contours, key=cv2.contourArea)        #获取最大轮廓
    x, y, w, h = cv2.boundingRect(largest_contour)              #获取最小外接矩形
    image = image[y : y + h, x : x + w]                         #切取最小外接矩形
    return image


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
        ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    elif type(image_path) is np.ndarray:
        img = image_path
    else:
        return None, None, None
    height, width = img.shape[:2]
    
    # 将角度转换为弧度
    angle_rad = math.radians(angle)
    
    """
    A = [a00 a01 tx]
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
    新画布左上角还是原点,画布中心为: (new_width/2, new_height/2)
    新图像的中心为: A*(width/2, height/2)^T
    水平方向移动单位: new_width/2 - (a00, a01)*(width/2, height/2)^T
    垂直方向移动单位: new_height/2 - (a10, a11)*(width/2, height/2)^T
    """
    tx = (new_width - width * a00 - height * a01) / 2
    ty = (new_height - width * a10 - height * a11) / 2
    
    # 构建完整的变换矩阵
    M = np.float32([[a00, a01, tx],
                    [a10, a11, ty]])
    
    # 应用变换到新的大小
    transformed_img = cv2.warpAffine(img, M, (new_width, new_height))
    
    return img, transformed_img, M

def add_cross_point(mat: np.ndarray, img: np.ndarray, point: int, linenum: int, threshold: int):
    """
    brief:  Get a cross point in the designated direction, determine which part it belongs to and count it.
    param:  @mat: A mat to record how many points fall in the three parts respectively.
            @point: The horizontal or vertical coordinate of the pixel point.
            @linenum: 
            @threshold: Used to divide the middle area.
    return: None.
    """
    linecls = linenum // 3
    length = img.shape[linecls]      #先对垂直线从左到右编号，再对水平线从下到上编号
    if point < length / 2 - threshold:
        mat[linenum][0] += 1
    elif point > length / 2 + threshold:
        mat[linenum][2] += 1
    else:
        mat[linenum][1] += 1

def get_matched_template(templates: list[np.ndarray], to_match: np.ndarray):
    """
    brief:  Compute the deviation between template and a mat gonna match with the template.\\
    param:  @template: A template to be matched with.\\
            @to_match: A mat to match with the above template.\\
    return: Deviation.
    """
    min_dev = (1 << 31) - 1
    """
    [3]                 [2]                     [1:0]
    是否有不同模板匹配     相同模板是否有误差相等的    最佳匹配模板号
    """
    matched_tpl = -1
    matched_more = []
    for i in range(0, 4):
        for j in range(0, 9):
            dev = sum(sum((templates[i][j] - to_match) ** 2))
            if dev < min_dev: 
                min_dev = dev
                matched_tpl = i
                matched_more.clear()
                # print("least dev", dev, '\t', 'dig:', j + 1)
            elif dev == min_dev:
                if (matched_tpl & 3) == i:
                    matched_tpl |= (1 << 2)
                    # print("1match tpl-digit:", i, '-', j + 1)
                else:
                    matched_tpl |= (1 << 3)
                    if matched_tpl & (1 << 3):
                        matched_more.append(i)
                    # print("2match tpl-digit:", i, '-', j + 1)
    return matched_tpl, matched_more


def rotation_angle(image_mats: list[np.ndarray]):
    """
    brief:  Decide what angle in degree the list of images need to rotate.
    param:  @image_mats: List of images of single digit, which must be listed in order.
    return: Angle in degree.

    note:   @image_mats must contain 9 matrices in order of 1 to 9, if no, replace it with None
    """
    for image_mat in image_mats:
        pixel_cnt = 0                                   #穿过的连通域的计数器
        space_cnt = 0                                   #穿过的空白区域计数器
        scores = [0 for i in range(0, 4)]

        """保存交点位置"""
        cross_points = np.int32([[0 for i in range(0, 3)] for i in range(0, 6)])
        
        image_mat = crop_bounding_rect(image_mat)
        height, width = image_mat.shape[0:2]
        hlines = (height // 4, height // 2, height * 3 // 4)
        vlines = (width // 4, width // 2, width * 3 // 4)

        cv2.line(image_mat, (vlines[0], 0), [vlines[0], height], 127)
        cv2.line(image_mat, (vlines[1], 0), [vlines[1], height], 127)
        cv2.line(image_mat, (vlines[2], 0), [vlines[2], height], 127)
        cv2.line(image_mat, (0, hlines[0]), [width, hlines[0]], 127)
        cv2.line(image_mat, (0, hlines[1]), [width, hlines[1]], 127)
        cv2.line(image_mat, (0, hlines[2]), [width, hlines[2]], 127)

        thick_uppper = int(3 / 5 * height)              #线宽上阈值，超过该值将线视为两个交点
        thick_bottom = int(1 / 40 * height)             #线宽下阈值，低于该值不被视为交点
        mid_treshold = int(1 / 6 * height)              #距离中点距离小于该值被视为中点交点
        sepspace = int(1 / 50 * height)                 #穿过的空白区域小于该阈值视为连通

        #垂直线
        for i in range(0, len(vlines)):
            old = 0
            for j in range(0, height): 
                if image_mat[j][vlines[i] - 1] == 0:    #与数字相交
                    if 0 < space_cnt < sepspace and old != 0:
                        pixel_cnt += space_cnt
                    space_cnt = 0

                    if pixel_cnt == 0:
                        old = j
                    pixel_cnt += 1            
                    
                    if j == height - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, image_mat, (j + old)/2, i, mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, image_mat, old, i, mid_treshold)
                            add_cross_point(cross_points, image_mat, j, i, mid_treshold)
                        pixel_cnt = old = 0
                else :                                  #与数字不相交
                    space_cnt += 1
                    if space_cnt >= sepspace or j == height - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, image_mat, (j + old)/2, i, mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, image_mat, old, i, mid_treshold)
                            add_cross_point(cross_points, image_mat, j, i, mid_treshold)
                        pixel_cnt = old = 0
        space_cnt = pixel_cnt = 0

        #如果采用digit_sharpening函数对图像进行了处理，这里就可以注释掉
        thick_uppper = int(3 / 5 * width)            #线宽上阈值，超过该值将线视为两个交点
        thick_bottom = int(1 / 40 * width)           #线宽下阈值，低于该值不被视为交点
        mid_treshold = int(1 / 6 * width)            #距离中点距离小于该值被视为中点交点
        sepspace = int(1 / 50 * width)               #穿过的空白区域小于该阈值视为连通

        #水平线
        for i in range(0, len(hlines)):
            old = 0
            for j in range(0, width): 
                if image_mat[hlines[i] - 1][j] == 0:                        #与数字相交
                    if 0 < space_cnt < sepspace and old != 0:
                        pixel_cnt += space_cnt
                    space_cnt = 0

                    if pixel_cnt == 0:
                        old = j 
                    pixel_cnt += 1

                    if j == width - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, image_mat, (j + old)/2, i + len(vlines), mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, image_mat, old, i + len(vlines), mid_treshold)
                            add_cross_point(cross_points, image_mat, j, i + len(vlines), mid_treshold)
                        pixel_cnt = old = 0
                else :                                                      #与数字不相交
                    space_cnt += 1
                    if space_cnt >= sepspace or j == width - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, image_mat, (j + old)/2, i + len(vlines), mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, image_mat, old, i + len(vlines), mid_treshold)
                            add_cross_point(cross_points, image_mat, j, i + len(vlines), mid_treshold)
                        pixel_cnt = old = 0

        print(cross_points)
        return image_mat

        tpl_idx, more = get_matched_template(templates, cross_points)
        if tpl_idx & (1 << 3):
            scores[tpl_idx & 3] -= 1
            for i in more:
                scores[i] -= 1
        elif tpl_idx & (1 << 2):
            scores[tpl_idx & 3] += 1
        else:
            scores[tpl_idx] += 3

    max_score = max(scores)
    if max_score > len(image_mats):
        best_match = scores.index(max_score)
        return (360 - best_match * 90) % 360
    else:
        pass

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "./images/digit/dig_{}.png".format(digit)
    plt.figure(figsize=(14, 6))

    """对正向的图进行旋转"""
    dummy1, image_mat, dummy2 = center_affine_transform(
        image_path, angle=angle, scale=1.0
    )

    image_mat = cv2.bitwise_not(image_mat)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(50,50))
    image_mat = cv2.erode(image_mat, kernel, iterations=1)
    image_mat = cv2.bitwise_not(image_mat)

    # plt.subplot(1, 2, 1)
    # plt.imshow(image_mat)
    # plt.title("Original Image")

    # print(rotation_angle([image_mat]))
    img = rotation_angle([image_mat])

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title("cropped Image")
    
    plt.tight_layout()
    plt.show()
