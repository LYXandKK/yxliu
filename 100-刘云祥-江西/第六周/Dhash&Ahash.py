import cv2
import numpy as np


# 均值哈希算法
def Ahash(GrayImg):
    Img = cv2.resize(GrayImg, (8, 8), interpolation=cv2.INTER_CUBIC)
    Flag = np.zeros([64, 1])
    SumValue = 0
    for i in range(8):
        for j in range(8):
            SumValue += Img[i, j]
    AveValue = SumValue / 64
    for i in range(8):
        for j in range(8):
            if (Img[i, j] > AveValue):
                Flag[i * 8 + j] = 1
    return Flag


# 差值哈希算法
def Dhash(GrayImg):
    Img = cv2.resize(GrayImg, (9, 8), interpolation=cv2.INTER_CUBIC)
    Flag = np.zeros([64, 1])
    SumNum = 0
    for i in range(8):
        for j in range(8):
            if Img[i, j] > Img[i, j + 1]:
                Flag[i * 8 + j] = 1
    return Flag


# 哈希值对比
def CmpHash(Hash1, Hash2):
    n = 0
    if len(Hash1) != len(Hash2):
        return -1
    for i in range(len(Hash1)):
        if Hash1[i] != Hash2[i]:
            n += 1
    return n


if __name__ == '__main__':
    Img1 = cv2.imread('HomeworkImg.jpg')
    Img2 = cv2.imread('HomeworkImg_rotate.jpg')
    GrayImg1 = cv2.cvtColor(Img1, cv2.COLOR_BGR2GRAY)
    GrayImg2 = cv2.cvtColor(Img2, cv2.COLOR_BGR2GRAY)
    # 均值哈希算法
    Flag1 = Ahash(GrayImg1)
    Flag2 = Ahash(GrayImg2)
    n1 = CmpHash(Flag1, Flag2)
    # 差值哈希算法
    Flag3 = Dhash(GrayImg1)
    Flag4 = Dhash(GrayImg2)
    n2 = CmpHash(Flag3, Flag4)
    print(n1, n2)
