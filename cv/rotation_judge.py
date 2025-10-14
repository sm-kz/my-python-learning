import numpy as np


forward_template = np.int32([[[0, 1, 0], [1, 0, 1]], [[0, 0, 1], [1, 0, 2]], [[0, 0, 1], [1, 1, 1]],
                            [[1, 0, 1], [1, 0, 1]], [[1, 0, 1], [2, 0, 1]], [[0, 1, 0], [1, 1, 1]],
                            [[0, 0, 1], [1, 0, 1]], [[0, 1, 0], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]]])

nforward_template = np.int32([[[1, 0, 1], [0, 1, 0]], [[1, 0, 2], [1, 0, 0]], [[1, 1, 1], [1, 0, 0]],
                             [[1, 0, 1], [1, 0, 1]], [[2, 0, 1], [1, 0, 1]], [[1, 1, 1], [0, 1, 0]],
                             [[1, 0, 1], [1, 0, 0]], [[1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 0, 1]]])


HORIZONTAL = 0
VERTICAL = 1

def add_cross_point(mat: np.ndarray, img: np.ndarray, point: int, dir: int, threshold: int):
    """
    brief:  Get a cross point in the designated direction, determine which part it belongs to and count it.
    param:  @mat: A mat to record how many points fall in the three parts respectively.
            @point: The horizontal or vertical coordinate of the pixel point.
            @dir: Can be @HORIZONTAL or @VERTICAL, the direction of the line's extension.
            @threshold: Used to divide the middle area.
    return: None.
    """
    if dir == VERTICAL or dir == HORIZONTAL:
        length = img.shape[0 if dir else 1]
        if point < length / 2 - threshold:
            mat[dir][0] += 1
        elif point > length / 2 + threshold:
            mat[dir][2] += 1
        else:
            mat[dir][1] += 1

def get_deviation(template: np.ndarray, to_match: np.ndarray, unmask: list[bool]):
    """
    brief:  Compute the deviation between template and a mat gonna match with the template.
    param:  @template: A template to be matched with.
            @to_match: A mat to match with the above template.
            @unmask: Indicate what digits don't count by setting corresponding items(say unmask[0]=1 to ignore 1).
    return: Deviation.
    """
    i = 0
    for li in unmask:
        if li:
            to_match[i] = template[i]
        i += 1
    return sum(sum(sum((template - to_match) ** 2)))

def rotation_angle(image_mats: list[np.ndarray]):
    """
    brief:  Decide what angle in degree the list of images need to rotate.
    param:  @image_mats: List of images of single digit, which must be listed in order.
    return: Angle in degree.

    note:   @image_mats must contain 9 matrices in order of 1 to 9, if no, replace it with None
    """
    digit = 0                                   #当前处理的数字
    count = 0                                   #穿过的连通域的计数器
    score = 0                                   #正表示需要与正向模板匹配，负表示需要与非正向模板匹配
    k = 1                                       #如果传入的样本过少，需要调节变量tolerance
    ignored_digit = [0 for i in range(0, 9)]    #屏蔽对应位的数字，从1开始

    """保存交点位置"""
    cross_points = np.int32([[[0 for i in range(0, 3)] for i in range(0, 2)] for i in range(0, 9)])
    
    for image_mat in image_mats:
        digit += 1
        if image_mat is None:
            k *= 0.85               
            ignored_digit[digit - 1] = 1        #数字图不存在，忽视
            continue

        height, width = image_mat.shape[0:2]
        hline = int(height / 2)
        vline = int(width / 2)

        """adjust parameters here"""
        thick_uppper = int(1 / 2 * (height + width) / 2)            #线宽上阈值，超过该值将线视为两个交点
        thick_bottom = int(1 / 20 * (height + width) / 2)           #线宽下阈值，低于该值不被视为交点
        mid_treshold = int(1 / 15 * (height + width) / 2)           #距离中点距离小于该值被视为中点交点
        tolerance = 3                                               #差值容忍度
        
        #水平线
        for i in range(1, width):                                   #旋转后莫名其妙边缘会出现黑边，适当改变起始值和结束值对结果无影响
            if image_mat[hline - 1][i] == 0:                        #与数字相交
                if count == 0:
                    old = i             
                count += 1
            else :                                                  #与数字不相交
                if count != 0:
                    if thick_bottom <= count <= thick_uppper:         #穿过的连通区域在两阈值之间，被视为一个点
                        add_cross_point(cross_points[digit - 1], image_mat, (i + old)/2, HORIZONTAL, mid_treshold)
                    else :                                          #穿过的连通区域大于上阈值，被视为两个点
                        add_cross_point(cross_points[digit - 1], image_mat, old, HORIZONTAL, mid_treshold)
                        add_cross_point(cross_points[digit - 1], image_mat, i, HORIZONTAL, mid_treshold)

                    count = 0
        count = 0

        if sum(cross_points[digit - 1][0]) >= 3:                    #水平线交点大于3
            score -= 1

        #垂直线
        for i in range(1, height):
            if image_mat[i][vline - 1] == 0:                        #与数字相交
                if count == 0:
                    old = i
                count += 1
            else :                                                  #与数字不相交
                if count != 0:
                    if thick_bottom <= count <= thick_uppper:     #穿过的连通区域小于阈值，被视为一个点
                        add_cross_point(cross_points[digit - 1], image_mat, (i + old)/2, VERTICAL, mid_treshold)
                    else :                                          #穿过的连通区域大于阈值，被视为两个点(一条线)
                        add_cross_point(cross_points[digit - 1], image_mat, old, VERTICAL, mid_treshold)
                        add_cross_point(cross_points[digit - 1], image_mat, i, VERTICAL, mid_treshold)
                    count = 0

        if sum(cross_points[digit - 1][1]) >= 3:                    #垂线交点大于3
            score += 1

    # print(cross_points)
    # print('score:', score)
    """
    理想情况下,数字正向时,与垂线交点为3的数字为6个,而水平线为0
    最差情况下,与垂线交点为3的数字为5个,而水平线为2
    """
    if score >= round(2 * k):               #正向或倒向
        deviation = get_deviation(forward_template, cross_points, ignored_digit)
        # print('deviation1:', deviation)
        if deviation <= tolerance * k:
            return 0
        else:
            return 180
    elif score <= -round(2 * k):            #数字横躺
        deviation = get_deviation(nforward_template, cross_points, ignored_digit)
        # print('deviation2:', deviation)
        if deviation <= tolerance * k:
            return 90
        else:
            return -90