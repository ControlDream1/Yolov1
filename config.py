import os
import torchvision.transforms as T
from sympy.abc import epsilon

# 超参数

data_path= 'data'


bath_size=32
epochs=100
warm_epochs=0
learning_rate=1E-4


Epsilon =1E-6  # 防止除零错误 说白了就是让那些很小的数不至于等于零  防止出现 数值不稳定 梯度爆炸和梯度消失

image_size=(448,448)    # yolo 的输入图像大小是 448 448


S=7
B=2
C=20   # 数据集一共多少个类别





