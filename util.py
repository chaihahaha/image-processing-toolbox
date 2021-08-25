import cv2
import numpy as np
from skimage.transform import resize

def resize_scaling(img, ratio):
    return resize(img, (int(img.shape[0]*ratio),int(img.shape[1]*ratio)))

def draw_bbox(img_path,label_path,result_path):
    # 读取图片(img_path)和标签(label_path)并绘制bbox，将结果写入到result_path
    img=cv2.imread(img_path)
    with open(label_path,"r") as f:
        label_string=f.read()
    label = [[int(j) for j in filter(lambda x: x !="",i.split(" "))] for i in label_string.split("\n")][1:-1]
    for bbox in label:
        #bbox=x12y12_to_xywh(i[0],i[1],i[2],i[3])
        img=cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 4)
    cv2.imwrite(result_path,img)


def x12y12_to_xywh(x1,y1,x2,y2):
    # bbox格式转换
    x,y=x1,y1
    w=x2-x1
    h=y2-y1
    return x,y,w,h

def white_balance(img):
    # 白平衡
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
def histeq(img):
    # 直方图均衡化
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output
def histogram_equalize(img):
    # 直方图均衡化
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))
def adjust_gamma(image, gamma=1.0):
    # 伽玛校正
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)
def flood_fill(img, xsize, ysize, x_start, y_start, color, cond):
    # 满足cond条件时就一直用color洪水填充，起始位置(x_start, y_start)，输入img为形状为(xsize, ysize, 3)的numpy array

    # 填充集s内的像素会被填充
    s = { (x_start, y_start) }
    # 完成集filled内的像素不会再次填充
    filled = set()
    while s:
        (x, y) = s.pop()
        # 如果满足填充条件cond且未被填充过，则填充并将周围像素加入填充集s
        if cond(img[x,y]) and (x, y) not in filled:
            # 填充并加入完成集filled
            img[x,y] = color
            filled.add((x, y))
            # 将周围像素加入填充集s
            if x > 0:
                s.add((x-1, y))
            if x < xsize - 1:
                s.add((x+1, y))
            if y > 0:
                s.add((x, y-1))
            if y < ysize - 1:
                s.add((x, y+1))
