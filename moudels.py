import torch
import torchvision
import torch.nn as nn
from PIL.FontFile import puti16

import config

class Yolov1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth=config.B*5+config.C

        layers=[
            # conv   1
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),


            # conv 2

            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),


            # conv  3
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,padding=2)
        ]


        # conv  4
        for  i in range(4):
            layers+=[
                nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
                nn.LeakyReLU()
           ]
        layers+=[
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]

        # conv 5

        for i in range(2):
            layers+=[
                nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        layers+=[
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        ]

        # conv 6

        for i in range(2):
            layers+=[
                nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]


        layers+=[
            nn.Flatten(),
            nn.Linear(7*7*1024,4096),         # linear 1
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(),
            nn.Linear(4096,config.S*config.S*self.depth)   # linear 2
        ]

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0),config.S,config.S,self.depth)
        )



