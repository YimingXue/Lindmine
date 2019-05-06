import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from config import config
import os
from skimage import io
from CBLoss import CBLoss

class ResNetv3_CBLoss(nn.Module):
    def __init__(self, config):
        super(ResNetv3_CBLoss, self).__init__()
        self.config = config
        self.input_conv = self.config.band
        self.input_fc = 1024 * (math.ceil((math.floor((config.patch_size-7+2*3)/2+1)-3+2*1)/2+1))**2
        # assert (math.ceil((math.floor((config.patch_size-7+2*3)/2+1)-3+2*1)/2+1)) == 8

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_conv, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
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
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024)
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
        c1 = self.conv(x) 
        b1 = self.relu(self.block1(c1))
        b2 = self.relu(self.block2(b1))
        b3 = self.relu(self.block3(b2))
        # b1 = self.relu(self.block1(c1) + self.downsample1(c1))
        # b2 = self.relu(self.block2(b1) + self.downsample2(b1))
        # b3 = self.relu(self.block3(b2) + self.downsample3(b2))
        f1 = b3.view(b3.size(0), -1)
        prob = self.fc(f1)

        softmax_prob = self.Softmax(prob)
            
        return prob, softmax_prob

    def calculate_objective(self, images, labels):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate focal loss
        loss = CBLoss(gamma=config.CBLoss_gamma, alpha=None, size_average=True)
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
    net = ResNetv2(config)
    print(net)