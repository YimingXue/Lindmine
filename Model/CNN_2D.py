import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from config import config
import os
from skimage import io

class CNN_2D(nn.Module):
    def __init__(self, config):
        super(CNN_2D, self).__init__()
        self.config = config
        self.input_fc1_nc = 128

        self.conv1 = nn.Sequential(
        # conv1
        nn.Conv2d(1, 32, 4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
        # conv2
        nn.Conv2d(32, 64, 5),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.MaxPool2d(2, stride=2)
        )

        self.conv3 = nn.Sequential(
        # conv3
        nn.Conv2d(64, 128, 4),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.MaxPool2d(2, stride=2)                                                                 
        )

        self.fc1 = nn.Sequential(
        # fc1
        nn.Linear(self.input_fc1_nc, config.num_classes),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True)
        )

        self.Softmax = nn.Softmax()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        f0 = c3.view(c3.size(0), -1)
        prob = self.fc1(f0)

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
    net = C3F4_CNN(config)
    print(net)