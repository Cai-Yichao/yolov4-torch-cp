import cv2
import numpy as np


# 尺度变换，两个尺度：1/2和1/4
def scale_transform(img, scale=0.5):
    dst = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return dst


# 图像对比度亮度变换 g(x,y) = a*f(x,y)+b -> α 调节对比度,一般建议取值(0.0~3.0)， β调节亮度
def bright_transform(img, alpha=1.0, beta=75):
    dst = img.copy()
    color = dst*alpha + beta
    color[color < 0] = 0
    color[color > 255] = 255
    dst = color.astype(np.uint8)
    return dst


def flip_transform(img, flip_param=0):
    '''
    参数2 必选参数。用于指定镜像翻转的类型，其中0表示绕×轴正直翻转，即垂直镜像翻转；1表示绕y轴翻转，即水平镜像翻转；
    -1表示绕×轴、y轴两个轴翻转，即对角镜像翻转。
    参数3 可选参数。用于设置输出数组，即镜像翻转后的图像数据，默认为与输入图像数组大小和类型都相同的数组。
    '''
    dst = cv2.flip(img, flip_param)
    return dst


def rotate_transform(img):
    dst = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return dst


if __name__ == '__main__':
    img = cv2.imread('data/src/1.jpg')
    dst = scale_transform(img)
    save_path = 'data/save/random/scale_1.jpg'
    cv2.imwrite(save_path, dst)
    print('1')
    dst = bright_transform(img)
    save_path = 'bright_upper.jpg'
    cv2.imwrite(save_path, dst)
    print('2')
    dst = bright_transform(img, beta=-75)
    save_path = 'bright_lower.jpg'
    cv2.imwrite(save_path, dst)
    flip_param = 0
    dst = flip_transform(img, flip_param)
    save_path = 'vertical.jpg'
    cv2.imwrite(save_path, dst)
    flip_param = 1
    dst = flip_transform(img, flip_param)
    save_path = 'horizon.jpg'
    cv2.imwrite(save_path, dst)
    dst = rotate_transform(img)
    save_path = 'rotate.jpg'
    cv2.imwrite(save_path, dst)
