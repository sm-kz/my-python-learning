import numpy as np
import cv2

def neighbours(x, y, image):
    """
    brief:  获取指定坐标的像素的八个相邻的像素值
    param:  @x: x坐标
            @y: y坐标
            @image: 输入的图像矩阵
    return: 保存相邻的八个像素值的列表
    """
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9


def transitions(neighbours):
    """
    brief:  以特定顺序,从0到1的变换出现的次数
    param:  @neighbours: neighbours()函数的返回值
    return: 从0到1的变换出现的次数
    """
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangSuen(image):
    """
    brief:  Zhang-Suen细化算法
    param:  @image: 二值图像, 前景像素值为1, 背景像素值为0
    return: 细化后的图像
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


def enhanced_upscale(image, target_size=(100, 100)):
    """
    brief:  增强型放大
    param:  @image: 需要放大的图像矩阵
            @target_size: 目标大小
    return: 放大后的图像矩阵
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
    brief:  多尺度融合放大
    param:  @image: 需要放大的图像矩阵
            @target_size: 目标大小
    return: 放大后的图像矩阵
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


def denoising(image, area_threshold=60, kernel_length=3):
    """
    brief:  保留图像连通性去噪、去毛刺
    param:  @image: 传入的图像矩阵, 必须进行归一化--前景为1、背景为0
            @area_threshold: 保留的连通区域面积阈值、 低于该阈值被消去
            @kernel_length: 垂直和水平去毛刺卷积核的长度
    return: 去噪后的图像矩阵
    Note:   @area_threshold不宜过小、 @kernel_length不宜过大
    """
    #开运算去除孤立点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
    
    #闭操作连接断开处
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_close)
    
    #获取八个方向的连通性
    num_labels_new, labels_new, stats_new, dummy = cv2.connectedComponentsWithStats(denoised, connectivity=8)
    
    denoised_img = np.zeros_like(image)
    for i in range(1, num_labels_new):
        area = stats_new[i, cv2.CC_STAT_AREA]
        if area >= area_threshold:              #保留面积大于阈值或与原始大组件对应的组件, 调整这个阈值
            denoised_img[labels_new == i] = 1
    
    #针对水平方向的毛刺(垂直核)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    opened_vertical = cv2.morphologyEx(denoised_img, cv2.MORPH_OPEN, kernel_vertical)
    
    #针对垂直方向的毛刺(水平核)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    opened_horizontal = cv2.morphologyEx(denoised_img, cv2.MORPH_OPEN, kernel_horizontal)
    
    #合并结果 - 只保留在两个方向上都存在的结构
    denoised_img = cv2.bitwise_and(opened_vertical, opened_horizontal)
    
    #用小闭运算填充可能的断裂
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final = cv2.morphologyEx(denoised_img, cv2.MORPH_CLOSE, kernel_small)
    
    return final


def laplacian_var(image: str | cv2.typing.MatLike):
    """
    brief:  使用Laplacian梯度方差评估图像清晰度值越大, 图像越清晰
    param:  @image: 传入的图像路径
    return: 归一化的图像质量评分
    """

    #图像质量对照表
    image_quality = {
        'Low':      1000,
        'Middle':   10000,
        'High':     20000,
        'SuperHigh':30000
    }

    if type(image) is str:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else: 
        img = image
    
    # 应用Laplacian算子并计算方差
    laplacian = cv2.Laplacian(img, cv2.CV_16U).var()
    
    #判断清晰度等级
    if  laplacian > image_quality['SuperHigh']:
        return 1
    elif laplacian > image_quality['High']:
        return 0.8
    elif laplacian > image_quality['Middle']:
        return 0.6
    elif laplacian > image_quality['Low']:
        return 0.4
    else:
        return 0.2
    

def thickening(thin_img, ksize=(5, 5), iterations=2, preserve_features=True):
    """
    brief:  对细化后的数字进行加粗
    param:  @thin_img: 细化后的图像、前景为1、背景为0
            @ksize: 采用膨胀卷积核的大小
            @iterations: 膨胀次数
            @preserve_features: 是否保留原有特征
    return: 加粗后的图像矩阵
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

def digit_process(image: str | cv2.typing.MatLike):
    """
    brief:  对尺寸小且模糊的数字图像进行处理、 返回尺寸大且较为清晰的数字图像
    param:  @image: 输入的图像为灰度图或二值图矩阵, 或任意类型图像的路径
    return: 返回图像为0, 1化的矩阵.前景为文字像素, 值为1
    """
    if type(image) is str:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else: 
        img = image
    img = enhanced_upscale(img, (100, 100))                     #放大模糊图像,目标尺寸太大会增加细化时间
    
    #根据图像质量进行二值化
    img_quality = laplacian_var(image)
    th, img = cv2.threshold(img, 127 * (2 - img_quality), 255, cv2.THRESH_BINARY)

    img = img < th                                              #01化并反转前景和背景
    img = denoising(img.astype(np.uint8), 80)                   #去噪
    img = zhangSuen(img)                                        #细化
    img = img.astype(np.uint8)                                  #转换像素类型
    img = multi_scale_upscale(img, (500, 500))                  #第二次放大
    img = thickening(img, iterations=2)                         #加粗前景
    return img                                                  #反转前景和背景


def digits_split(image: cv2.typing.MatLike, precut_edge: int = 0)->list[cv2.typing.MatLike]:
    """ 
    brief:  将多位数字图像进行分割
    param:  @image: 输入的图像为灰度图或二值图矩阵
            @precut_edge: 处理图像前, 对图像边框进行裁剪的大小
    return: 返回被分割的图像列表
    """
    height, width = image.shape[:2]
    image = image[precut_edge : height - precut_edge, precut_edge : width - precut_edge]
    threshold = (height + width) // 50                  #设置阈值避免噪声干扰

    def continuous_segments(projection):
        in_digit = False                                #表明是否在数字内
        segments = []                                   #记录每一段的位置
        nonlocal threshold
        for i, count in enumerate(projection):
            if count >= 0 and not in_digit:
                in_digit = True
                start = i
            elif count == 0 and in_digit:
                in_digit = False
                segments.append((start, i))
            
        if in_digit:                                    #遍历结束后还在数字内
            segments.append((start, len(projection) - 1))

        #过滤掉连续长度小于阈值的段
        segments = list(filter(lambda seg: seg[1] - seg[0] >= threshold, segments))
        return segments

    vprojection = np.sum(image, axis=0)                 #计算每列垂直投影像素(1)和
    seg = continuous_segments(vprojection)
    #垂直方向扫描到多个数字
    if len(seg) > 1:                                   
        digits = []
        for dig_start, dig_end in seg:
            digit_image = image[:, dig_start : dig_end]
            digits.append(digit_image)
    
        return digits
    
    hprojection = np.sum(image, axis=1)                 #计算每行水平投影像素(1)和
    seg = continuous_segments(hprojection)
    #水平方向扫描到多个数字
    if len(seg) > 1:                                   
        digits = []
        for dig_start, dig_end in seg:
            digit_image = image[dig_start : dig_end, :]
            digits.append(digit_image)
    
        return digits
    return [image]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = cv2.imread('./images/dtcdig/77/77.jpg_cropped_3.jpg')
    score = laplacian_var(img)
    print(f"Laplacian清晰度得分: {score}")
