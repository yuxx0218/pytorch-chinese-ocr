# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, imgC=1):
        super(CNN, self).__init__()

        padding = 3 // 2  # darknet padding=1 means pytorch padding=k_size//2
        self.conv1 = nn.Conv2d(imgC, 64, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.mpool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        padding = 3 // 2
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding)
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        padding = 3 // 2
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding)
        self.relu3 = nn.ReLU()
        padding = 3 // 2
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding)
        self.relu4 = nn.ReLU()
        self.mpool3 = nn.MaxPool2d(2, (2, 1), 0, ceil_mode=True)
        padding = 3 // 2
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding)
        self.relu5 = nn.ReLU()
        padding = 3 // 2
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding)
        self.relu6 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(2, (2, 1), 0, ceil_mode=True)

        self.conv7 = nn.Conv2d(512, 512, 2, 1, 0)
        self.relu7 = nn.ReLU()
        padding = 1 // 2
        self.conv8 = nn.Conv2d(512, 11316, 1, 1, padding)

    def forward(self, x):
        x = self.mpool1(self.relu1(self.conv1(x)))
        x = self.mpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.mpool3(self.relu4(self.conv4(x)))
        x = self.relu5(self.conv5(x))
        x = self.mpool4(self.relu6(self.conv6(x)))
        x = self.relu7(self.conv7(x))
        x = self.conv8(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = F.log_softmax(x, dim=2)

        return x



if __name__ == '__main__':
    import time
    import torch

    data = torch.randn(1, 1, 32, 256)

    model = CNN(1)
    model.eval()
    print(model)

    start = time.process_time()
    out = model(data)
    end = time.process_time()
    print('Model Inference Time is {}'.format(end-start))
    print(out.size())

