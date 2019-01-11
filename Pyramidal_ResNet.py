import torch
import torch.nn as nn
from config import config
import math
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision

class Pyramidal_ResNet(nn.Module):
    def __init__(self, config):
        super(Pyramidal_ResNet, self).__init__()
        self.config = config
        self.mid_units = 64
        self.alpha = 50
        self.number_units = 9
        self.added_channel = math.floor(self.alpha/self.number_units)

        self.Input_module = nn.Sequential(
            nn.Conv2d(in_channels=self.config.band, out_channels=self.config.band, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.config.band),
        )

        self.Pyramidal_module_P1 = nn.Sequential(
            # B(1)_1
            nn.BatchNorm2d(self.config.band),
            nn.Conv2d(in_channels=self.config.band, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band),
            nn.ReLU(),
            
            # B(1)_2
            nn.BatchNorm2d(self.config.band),
            nn.Conv2d(in_channels=self.config.band, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel),
            nn.ReLU(),
            
            # B(1)_3
            nn.BatchNorm2d(self.config.band+self.added_channel),
            nn.Conv2d(in_channels=self.config.band+self.added_channel, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*2, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*2),
            nn.ReLU()
        )

        self.P1_ShortCut = nn.Sequential(
            nn.BatchNorm2d(self.config.band),
            nn.Conv2d(in_channels=self.config.band, out_channels=self.config.band+self.added_channel*2, kernel_size=1, stride=1)
        )

        self.Pyramidal_module_P2 = nn.Sequential(
            # B(2)_1
            nn.BatchNorm2d(self.config.band+self.added_channel*2),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*2, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*3, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            # B(2)_2
            nn.BatchNorm2d(self.config.band+self.added_channel*3),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*3, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*4, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*4),
            nn.ReLU(),
            
            # B(2)_3
            nn.BatchNorm2d(self.config.band+self.added_channel*4),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*4, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*5, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*5),
            nn.ReLU()
        )

        self.P2_ShortCut = nn.Sequential(
            nn.BatchNorm2d(self.config.band+self.added_channel*2),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*2, out_channels=self.config.band+self.added_channel*5, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.config.band+self.added_channel*5),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*5, out_channels=self.config.band+self.added_channel*5, kernel_size=3, stride=1)
        )        

        self.Pyramidal_module_P3 = nn.Sequential(
            # B(3)_1
            nn.BatchNorm2d(self.config.band+self.added_channel*5),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*5, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*6, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*6),
            nn.ReLU(),
            
            # B(3)_2
            nn.BatchNorm2d(self.config.band+self.added_channel*6),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*6, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*7, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*7),
            nn.ReLU(),
            
            # B(3)_3
            nn.BatchNorm2d(self.config.band+self.added_channel*7),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*7, out_channels=self.mid_units, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.mid_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.mid_units),
            nn.Conv2d(in_channels=self.mid_units, out_channels=self.config.band+self.added_channel*8, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.config.band+self.added_channel*8),
            nn.ReLU()
        )

        self.P3_ShortCut = nn.Sequential(
            nn.BatchNorm2d(self.config.band+self.added_channel*5),
            nn.Conv2d(in_channels=self.config.band+self.added_channel*5, out_channels=self.config.band+self.added_channel*8, kernel_size=3, stride=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.band+self.added_channel*8, self.config.indianPines_class),
        )
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Extract features
        x = self.Input_module(x)
        P1 = self.Pyramidal_module_P1(x)
        P1_ShortCut = self.P1_ShortCut(x)
        
        assert P1.shape == P1_ShortCut.shape
        P2_mid = P1 + P1_ShortCut
        
        P2 = self.Pyramidal_module_P2(P2_mid)
        P2_ShortCut = self.P2_ShortCut(P2_mid)
        
        assert P2.shape == P2_ShortCut.shape
        P3_mid = P2 + P2_ShortCut
        
        P3 = self.Pyramidal_module_P3(P3_mid)
        P3_ShortCut = self.P3_ShortCut(P3_mid)
        
        assert P3.shape == P3_ShortCut.shape
        H = P3 + P3_ShortCut

        # classification
        H = H.view(-1, self.config.band+self.added_channel*8)
        prob = self.classifier(H) 
        softmax_prob = self.Softmax(prob)
        return prob, softmax_prob
    
    def calculate_objective(self, images, labels):
        labels = torch.squeeze(labels)
        labels = labels.type(torch.cuda.LongTensor)
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
    
if __name__ == '__main__':
    model = Pyramidal_ResNet(config)
    print(model)

