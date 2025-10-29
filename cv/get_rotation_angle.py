import cv2
import numpy as np
import math
import os
from typing import Union
import blurred_digit_image_process as bdp

"""
下面是匹配模板, 对应不同角度(顺时针旋转)下的数字模板. 模板匹配采用了水平和垂直方向的线各三条, 模板的axis0表示每个数字的特征; 
axis1表示垂直线(前三, 从左到右)、水平线(后三, 从上到下)与数字的特征交点; axis2表示交点的位置左(上)/中/右(下)
"""
template_0 = np.int32([[[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],    #0
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1]],   #1
                        [[1, 0, 2], [1, 0, 2], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0]],   #2
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1]],   #3
                        [[0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1]],   #4!
                        [[1, 1, 1], [2, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]],   #5!
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1]],   #6!
                        [[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],   #7
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],   #8!
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]]])  #9!

template_90 = np.int32([[[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],   #0
                        [[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],   #1
                        [[1, 0, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [2, 0, 1], [1, 1, 1]],   #2
                        [[1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]],   #3
                        [[1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]],   #4
                        [[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 2], [1, 1, 1]],   #5
                        [[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],   #6
                        [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1]],   #7
                        [[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],   #8
                        [[0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])  #9

template_180 = np.int32([[[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],  #0
                        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1]],   #1
                        [[1, 1, 1], [2, 0, 1], [2, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]],   #2
                        [[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0]],   #3
                        [[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]],   #4
                        [[1, 1, 1], [1, 0, 2], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],   #5
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]],   #6
                        [[0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],   #7
                        [[1, 1, 1], [1, 1, 1], [1 ,1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],   #8
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1]]])  #9

template_270 = np.int32([[[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],  #0
                        [[1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],   #1
                        [[1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 2], [1, 0, 2]],   #2
                        [[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],   #3
                        [[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1]],   #4
                        [[0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [2, 0, 1], [1, 1, 1]],   #5
                        [[0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],   #6
                        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]],   #7
                        [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],   #8
                        [[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])  #9

templates = [template_0, template_90, template_180, template_270]

def crop_bounding_rect(image: np.ndarray, precut_edge: int = 0, bitwisenot=False, need01=False):
    """
    brief:  切取数字的最小外接矩形
    param:  @image:         要剪切的图像
            @precut_edge:   前置工作, 剪切图像边框大小
            @bitwisenot:    需要前景为高像素值
            @need01:        是否需要归一化
    return: 截取最小外接矩形后的图像
    """
    height, width = image.shape[:2]
    image = image[precut_edge : height - precut_edge, precut_edge : width - precut_edge]

    if need01:
        dummy, image = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY)  #可能需要调整二值阈值

    if bitwisenot:
        img_inv = cv2.bitwise_not(image)                                #颜色反转,识别边框需要白色在内部
    else:
        img_inv = cv2.bitwise_xor(image, np.zeros_like(image))

    contours, dummy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)       #获取轮廓
    largest_contour = max(contours, key=cv2.contourArea)        #获取最大轮廓
    x, y, w, h = cv2.boundingRect(largest_contour)              #获取最小外接矩形
    image = image[y : y + h, x : x + w]                         #切取最小外接矩形

    return image


def center_affine_transform(image_path: Union[str, np.ndarray], angle: float = 0, scale: float = 1.0):
    """
    brief:  以图像中线为原点进行旋转
    param:  @image_path: 图像路径或矩阵
            @angle: 顺时针旋转的角度
            @scale: 缩放倍数
    return: 1.原图
            2.旋转后的图
            3.仿射变换矩阵
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
    新画布左上角还是原点,画布中心为:   (new_width/2, new_height/2)
    新图像的中心为:                 A*(width/2, height/2)^T
    水平方向移动单位:               new_width/2 - (a00, a01)*(width/2, height/2)^T
    垂直方向移动单位:               new_height/2 - (a10, a11)*(width/2, height/2)^T
    """
    tx = (new_width - width * a00 - height * a01) / 2
    ty = (new_height - width * a10 - height * a11) / 2
    
    # 构建完整的变换矩阵
    M = np.float32([[a00, a01, tx],
                    [a10, a11, ty]])
    
    # 应用变换到新的大小
    transformed_img = cv2.warpAffine(img, M, (new_width, new_height))
    
    return img, transformed_img, M

def add_cross_point(mat: np.ndarray, size: np.array, point: int, linenum: int, threshold: int):
    """
    brief:  增加对应位置的交点
    param:  @mat:       保存特征交点的矩阵
            @size:      图像的尺寸, 第一个元素为高度, 第二个为宽度
            @point:     线上点的位置
            @linenum:   线编号, 从左到右, 从上到下
            @threshold: 中心位置两侧阈值
    return: None.
    """
    linecls = linenum // 3
    length = size[linecls]      #先对垂直线从左到右编号，再对水平线从下到上编号
    if point < length / 2 - threshold:
        mat[linenum][0] += 1
    elif point > length / 2 + threshold:
        mat[linenum][2] += 1
    else:
        mat[linenum][1] += 1

def get_matched_template(templates: list[np.ndarray], to_match: np.ndarray):
    """
    brief:  匹配模板
    param:  @template: 要匹配的模板
            @to_match: 要与模板匹配的矩阵
    return: 1.模板匹配数
            2.匹配的多的模板的
    """
    min_dev = (1 << 31) - 1
    """
    [3]                 [2]                     [1:0]
    是否有不同模板匹配     相同模板是否有误差相等的    最佳匹配模板号
    """
    matched_tpl = (1 << 4)
    matched_more = []
    #这里顺序为3、1、0、2表示270°匹配优先级最高、90°次之...
    for i in [3, 1, 0, 2]:
        for j in range(0, 10):
            dev = sum(sum((templates[i][j] - to_match) ** 2))
            if dev < min_dev: 
                min_dev = dev
                matched_tpl = i
                matched_more.clear()
            elif dev == min_dev:
                if (matched_tpl & 3) == i:                  #同一模板有多个匹配
                    matched_tpl |= (1 << 2)
                else:                                       #与多个不同的模板匹配
                    matched_more.append(matched_tpl & 3)
                    matched_tpl = (1 << 3) | i
    return matched_tpl, matched_more


def rotation_angle(image_mats: list[np.ndarray]):
    """
    brief:  获取数字图像所需旋转的角度. 6 8 9因特征原因无法判断如何旋转
    param:  @image_mats: 图像列表
    return: 旋转角度, 只能为90°的倍数
    """
    scores = [0 for i in range(4)]
    for image_mat in image_mats:
        pixel_cnt = 0                                   #穿过的连通域的计数器
        space_cnt = 0                                   #穿过的空白区域计数器

        #保存交点位置
        cross_points = np.int32([[0 for i in range(0, 3)] for i in range(0, 6)])
        #截取最小外接矩形
        image_mat = crop_bounding_rect(image_mat)   
        image_mat = (1 - image_mat) * 255               #去归一化

        height, width = image_mat.shape[0:2]
        hlines = (height // 4, height // 2, height * 3 // 4)
        vlines = (width // 4, width // 2, width * 3 // 4)

        # cv2.line(image_mat, (vlines[0], 0), [vlines[0], height], 127)
        # cv2.line(image_mat, (vlines[1], 0), [vlines[1], height], 127)
        # cv2.line(image_mat, (vlines[2], 0), [vlines[2], height], 127)
        # cv2.line(image_mat, (0, hlines[0]), [width, hlines[0]], 127)
        # cv2.line(image_mat, (0, hlines[1]), [width, hlines[1]], 127)
        # cv2.line(image_mat, (0, hlines[2]), [width, hlines[2]], 127)

        thick_uppper = int(1 / 2 * height)              #线宽上阈值，超过该值将线视为两个交点
        thick_bottom = int(1 / 40 * height)             #线宽下阈值，低于该值不被视为交点
        mid_treshold = int(1 / 14 * height)             #距离中点距离小于该值被视为中点交点
        sepspace = int(1 / 50 * height)                 #穿过的空白区域小于该阈值视为连通

        #垂直线
        for i in range(0, len(vlines)):
            old = 0
            for j in range(0, height): 
                if image_mat[j][vlines[i] - 1] == 0:                        #与数字相交
                    if 0 < space_cnt < sepspace and old != 0:               #两交点像素间的空白小于阈值,算作交点像素
                        pixel_cnt += space_cnt
                    space_cnt = 0

                    if pixel_cnt == 0:
                        old = j
                    pixel_cnt += 1            
                    
                    if j == height - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, (height, width), (j + old)/2, i, mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, (height, width), old, i, mid_treshold)
                            add_cross_point(cross_points, (height, width), j, i, mid_treshold)
                        pixel_cnt = old = 0
                else :                                                      #与数字不相交
                    space_cnt += 1
                    if space_cnt >= sepspace or j == height - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, (height, width), (j - space_cnt + old)/2, i, mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, (height, width), old, i, mid_treshold)
                            add_cross_point(cross_points, (height, width), j - space_cnt, i, mid_treshold)
                        pixel_cnt = old = space_cnt = 0

        thick_uppper = int(1 / 2 * width)            #线宽上阈值，超过该值将线视为两个交点
        thick_bottom = int(1 / 40 * width)           #线宽下阈值，低于该值不被视为交点
        mid_treshold = int(1 / 12 * width)           #距离中点距离小于该值被视为中点交点
        sepspace = int(1 / 50 * width)               #穿过的空白区域小于该阈值视为连通

        #水平线
        for i in range(0, len(hlines)):
            old = 0
            for j in range(0, width): 
                if image_mat[hlines[i] - 1][j] == 0:                        #与数字相交
                    if 0 < space_cnt < sepspace and old != 0:               #两交点像素间的空白小于阈值,算作交点像素
                        pixel_cnt += space_cnt
                    space_cnt = 0

                    if pixel_cnt == 0:
                        old = j 
                    pixel_cnt += 1

                    if j == width - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, (height, width), (j + old)/2, i + len(vlines), mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, (height, width), old, i + len(vlines), mid_treshold)
                            add_cross_point(cross_points, (height, width), j, i + len(vlines), mid_treshold)
                        pixel_cnt = old = 0
                else :                                                      #与数字不相交
                    space_cnt += 1
                    if space_cnt >= sepspace or j == width - 1:
                        if thick_bottom <= pixel_cnt <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                            add_cross_point(cross_points, (height, width), (j - space_cnt + old)/2, i + len(vlines), mid_treshold)
                        elif pixel_cnt > thick_uppper:                      #穿过的连通区域大于上阈值，被视为两个点
                            add_cross_point(cross_points, (height, width), old, i + len(vlines), mid_treshold)
                            add_cross_point(cross_points, (height, width), j - space_cnt, i + len(vlines), mid_treshold)
                        pixel_cnt = old = space_cnt = 0

        #计算得分
        tpl_idx, more = get_matched_template(templates, cross_points)
        if tpl_idx & (1 << 3):
            scores[tpl_idx & 3] += 1
            for i in more:
                scores[i] += 1
        elif tpl_idx & (1 << 2):
            scores[tpl_idx & 3] += 2
        else:
            scores[tpl_idx] += 4

    # print(cross_points)

    max_score = max(scores)
    best_match = scores.index(max_score)
    # print(f"max score: {max_score}, best match: {best_match}")
    # return (360 - best_match * 90) % 360, image_mat

    return (360 - best_match * 90) % 360

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_paths = ["./images/dtcdig/{0}/{0}.jpg_cropped_{1}.jpg".format(15, i) for i in (7,8,6)]
    plt.figure(figsize=(14, 6))

    image_mats = []
    for image_path in image_paths:
        img = bdp.digit_process(image_path)                  #处理图像

        """对正向的图进行旋转"""
        dummy1, image_mat, dummy2 = center_affine_transform(
            img, angle=270, scale=1.0
        )

        split = bdp.digits_split(image_mat, sum(img.shape[:2]) // 40)   #分割图像
        image_mats.extend(split)

    angle = rotation_angle(image_mats)
    print(angle)

    # plt.imshow(image_mat)
    # plt.title("cropped Image")
    
    # plt.tight_layout()
    # plt.show()
