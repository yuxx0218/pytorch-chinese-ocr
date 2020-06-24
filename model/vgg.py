# -*- coding: utf-8 -*-

import torch.nn as nn



class VGG(nn.Module):
    def __init__(self, imgC=3): # BGR
        super(VGG, self).__init__()

        padding = 3 // 2
        # block 1
        self.conv1_1 = nn.Conv2d(imgC, 64, 3, 1, 1)
        self.lrelu1_1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.lrelu1_2 = nn.LeakyReLU(negative_slope=0.1)

        self.mpool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.lrelu2_1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.lrelu2_2 = nn.LeakyReLU(negative_slope=0.1)

        self.mpool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.lrelu3_1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.lrelu3_2 = nn.LeakyReLU(negative_slope=0.1)

        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.lrelu3_3 = nn.LeakyReLU(negative_slope=0.1)

        self.mpool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lrelu4_1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu4_2 = nn.LeakyReLU(negative_slope=0.1)

        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu4_3 = nn.LeakyReLU(negative_slope=0.1)

        self.mpool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # block 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu5_1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu5_2 = nn.LeakyReLU(negative_slope=0.1)

        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu5_3 = nn.LeakyReLU(negative_slope=0.1)

        # rpn
        self.rpn6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.1)

        # out
        padding = 1 // 2
        self.out7 = nn.Conv2d(512, 40, 1, 1, 0)

    def forward(self, x):
        x = self.lrelu1_1(self.conv1_1(x))
        # print(x.size())
        x = self.lrelu1_2(self.conv1_2(x))
        # print(x.size())
        x = self.mpool1(x)
        # print(x.size())

        x = self.lrelu2_1(self.conv2_1(x))
        # print(x.size())
        x = self.lrelu2_2(self.conv2_2(x))
        # print(x.size())
        x = self.mpool2(x)
        # print(x.size())

        x = self.lrelu3_1(self.conv3_1(x))
        # print(x.size())
        x = self.lrelu3_2(self.conv3_2(x))
        # print(x.size())
        x = self.lrelu3_3(self.conv3_3(x))
        # print(x.size())
        x = self.mpool3(x)
        # print(x.size())

        x = self.lrelu4_1(self.conv4_1(x))
        # print(x.size())
        x = self.lrelu4_2(self.conv4_2(x))
        # print(x.size())
        x = self.lrelu4_3(self.conv4_3(x))
        # print(x.size())
        x = self.mpool4(x)
        # print(x.size())

        x = self.lrelu5_1(self.conv5_1(x))
        # print(x.size())
        x = self.lrelu5_2(self.conv5_2(x))
        # print(x.size())
        x = self.lrelu5_3(self.conv5_3(x))
        # print(x.size())

        x = self.lrelu6(self.rpn6(x))
        # print(x.size())
        x = self.out7(x)
        # print(x.size())

        return x



if __name__ in '__main__':
    import time
    import torch

    data = torch.randn(1, 3, 1800, 828).cuda()

    model = VGG(3).cuda()
    checkpoint = torch.load('../weight/text/text.pth.tar')
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    start = time.perf_counter()
    out = model(data)
    end = time.perf_counter()
    print('Model Inference Time is {}'.format(end - start))
    print(out.size())
