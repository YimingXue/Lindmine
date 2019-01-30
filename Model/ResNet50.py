import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0,'/home/xueyiming/DeepTransfer/')
from config import config
import os
from skimage import io

class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        self.config = config
        self.input_conv = self.config.band
        # self.input_fc = 2048 * (math.ceil((math.floor((config.patch_size-7+2*3)/2+1)-3+2*1)/2+1))**2
        assert (math.floor((math.floor((config.patch_size-7+2*3)/2+1)-3+2*1)/2+1)) == 8
        self.input_fc = 2048

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_conv, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.relu = nn.ReLU()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.block1mid = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.block2mid = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.block3mid = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.block4mid = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.fc = nn.Sequential(
        nn.Linear(self.input_fc, 2048),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 1024),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True),
        nn.Linear(1024, config.num_classes)
        )
        self.Softmax = nn.Softmax()

    def forward(self, x):
        c1 = self.conv(x) # NxCx8x8

        b1 = self.relu(self.block1(c1) + self.downsample1(c1)) # Nx256x8x8
        b1 = self.relu(self.block1mid(b1) + b1)
        b1 = self.relu(self.block1mid(b1) + b1)


        b2 = self.relu(self.block2(b1) + self.downsample2(b1)) # Nx512x4x4
        b2 = self.relu(self.block2mid(b2) + b2)
        b2 = self.relu(self.block2mid(b2) + b2)
        b2 = self.relu(self.block2mid(b2) + b2)

        b3 = self.relu(self.block3(b2) + self.downsample3(b2)) # Nx1024x2x2
        # b3 = self.relu(self.block3mid(b3) + b3)
        # b3 = self.relu(self.block3mid(b3) + b3)

        b4 = self.relu(self.block4(b3) + self.downsample4(b3)) # Nx2048x1x1

        f1 = b4.view(b4.size(0), -1)
        prob = self.fc(f1)

        softmax_prob = self.Softmax(prob)
            
        return prob, softmax_prob

    def calculate_objective(self, images, labels):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()
        output = loss(prob,labels)
        return output
    
    def calculate_classification_accuary(self, images, labels, accuracy_per_class, number_per_class):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)
        num_correct_classified = torch.sum(torch.eq(prediction, labels)).item()
        # calculate accuracy per class
        for i in range(len(labels)):
            if labels[i] == prediction[i]:
                accuracy_per_class[labels[i]] += 1
            number_per_class[labels[i]] += 1
        return num_correct_classified, accuracy_per_class, number_per_class
    
    def inference_classification(self, images):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)

        return prediction


if __name__ == "__main__":
    net = ResNet50(config)
    print(net)