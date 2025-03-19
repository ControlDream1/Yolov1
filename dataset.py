import os
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn as nn
from numpy.ma.core import negative
from sympy.physics.quantum.gate import normalized
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import utils
import tqdm

import config


class Yolov1_VOC_Dataset(Dataset):
    def __init__(self,set_type,normalize=None,augment=False):
        assert set_type in {'trian','test'}

        # 调用 torchvision 中专门为 voc 数据集准备的dataset

        self.dataset=VOCDetection(
            root="VOCdevkit/VOC2012",
            year="2012",
            image_set=('train' if set_type=='train'else'test'),  # 判断是train还是test  相当与选择是训练集还是测试集
            download=False,              # 选择是否下载 数据集  (如果本地已经下载好了无需重复下载
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.image_size)
            ])
        )
        self.normalize=normalize    # 归一化操作
        self.augment=augment        # 数据增强


        # 为不同类别生成字典  例如 car - 0
        self.classes = utils.load_class_dict()   # 方便后面统一类别索引 和 生成 独热编码


        # 生成类别字典  主要思想就是遍历dataset 遇到没遇到过的类别名称 就赋予index 同时index++
        index =0
        if len(self.classes)==0:
            for i , data_pair in enumerate(tqdm(self.dataset,desc="生成类别字典")):
                data ,label = data_pair
                for j , bbox_pair in enumerate(utils.get_bounding_boxs(label)):
                    name , coords=bbox_pair
                    if name not in self.classes:
                        self.classes[name]=index
                        index+=1

        utils.save_class_dict(self.classes)   #保存类别字典

    # getitem目的通常就是将数据集的 图片 和 标签 相关信息加工好后 return

    def __getitem__(self, i):
        data,label=self.dataset[i]
        original_data=data

        # 随机平移
        x_shift=int((0.2*random.random()-0.1)*config.image_size[0])
        y_shift=int((0.2*random.random()-0.1)*config.image_size[1])

        #随机缩放
        scale =1+0.2*random.random()

        # 如果进行数据增强的话
        if self.augment :
            # 应用平移和缩放   affine 函数的详细参数作用可以查阅资料
            data=TF.affine(data,angle=0.0,translate=[x_shift,y_shift],scale=scale,shear=[0.0])
            # 随机调整色调
            dara=TF.adjust_hue(data,hue_factor=0.2*random.random()-0.1)
            # 随机调整饱和度
            data=TF.adjust_saturation(data,saturation_factor=0.2*random.random()-0.1)

        # 如果进行归一化的话
        if self.normalize :
            data=TF.normalize(data,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

        # 至此 我们图片已经加工完成 可以return   接下来就是标签该怎么加工 如何return


        grid_size_x=data.size(dim=2)/config.S  # 每个 grid cell 的宽
        grid_size_y=data.size(dim=1)/config.S  # 每个 grid cell 的高

        boxes={}    #  记录每个格子已经使用的边框数
        class_names={}  # 记录每个格子的类别 避免冲突
        depth=config.B*5+config.C  # 特征向量深度  5个信息加上20种类别的概率

        ground_truth=torch.zeros((config.S,config.S,depth)) # 初始化ground_truth张量

        for j,bbox_pair in enumerate(utils.get_bounding_boxs(label)):   # 读取每张图片的标签信息
            name,coords =bbox_pair
            assert name in self.classes ,f"未找到相应类别名称'{name}'"
            class_index=self.classes[name]
            x_min,x_max,y_min,y_max=coords

            # 增强坐标    提高模型的泛化能力

            if self.augment:
                half_width=config.image_size[0]/2
                half_height=config.image_size[1]/2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_man = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift


            #计算 ground_truth 的中心坐标
            mid_x=(x_min+x_max)/2
            mid_y=(y_min+y_max)/2

            #同时我们要把GT的坐标映射到 grid cell 中去  看看他在具体S*S的哪一个图片块中
            col=int(mid_x//grid_size_x)
            row=int(mid_y//grid_size_y)

            if 0<=col<=config.S and 0<=row<=config.S:
                cell =(col,row)
                if cell not in class_names or name==class_names[cell]:

                    # 设置独热向量
                    one_hot=torch.zeros(config.C)
                    one_hot[class_index]=1.0

                    #将独热向量填充到 GT 中去 （值得注意的是  GT向量 和 特征向量不太一样  GT是前20通道为类别 用独热向量填充）
                    ground_truth[col,row,:config.C]=one_hot
                    class_names[cell]=name

                    #接着填充 GT的 (x,y,w,h,置信度) 五个值
                    bbox_index=boxes.get(cell,0)   # 当前是第几个框
                    if bbox_index <config.B:
                        bbox_truth=(

                            (mid_x-col*grid_size_x)/config.image_size[0],    # 计算中心坐标在当前grid cell  x方向的偏移量  除以图片的宽实现归一化
                            (mid_y-row*grid_size_y)/config.image_size[1],    # 同上
                            (x_max-x_min)/config.image_size[0],              # GT宽归一化到grid cell 中去
                            (y_max-y_min)/config.image_size[1],              # 同上
                            1.0                                              # 置信度
                        )

                    # 因为GT的形状和特征形状相同  预测时有B个框  但是GT只有一个正确值 所以我们重复填充GT(x,y,w,h,confidence,c) B次
                    bbox_start=5*bbox_index+config.C
                    ground_truth[row,col,bbox_start:]=torch.tensor(bbox_truth).repeat(config.B-bbox_index)
                    boxes[cell]=bbox_index+1


        return data,ground_truth,original_data        # 返回处理后的图片  填充完毕的GT特征  未经处理的图片



    def __len__(self):
        return len(self.dataset)



if __name__ == '__main__':
    #可视化数据集
    obj_classes=utils.load_class_array()
    train_set=Yolov1_VOC_Dataset('train',normalize=True,augment=True)

    negative_labels=0
    smallest=0
    largest=0
    for data,label,_ in train_set :
        negative_labels+=torch.sum(label<0).item()
        smallest=min(smallest,torch.min(data).item())
        largest=max(largest,torch.max(data).item())
        utils.plot_boxes(data,label,obj_classes,max_overlap=float('inf'))












