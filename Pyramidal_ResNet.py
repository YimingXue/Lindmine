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

        self.Input_module = nn.Sequential(
            nn.Conv2d(in_channels=self.config.indianPines_band, out_channels=self.config.indianPines_band, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(self.config.indianPines_band),
        )

        self.Pyramidal_module_P1 = nn.Sequential(
            # B(1)_1
            nn.BatchNorm2d(self.config.indianPines_band),
            nn.Conv2d(in_channels=self.config.indianPines_band, out_channels=64, kernel_size=(1,1), stride=(1,1)),
            # B(1)_2
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7), stride=(1,1), padding=(3,3)),
            # B(1)_3
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=self.config.indianPines_band, kernel_size=(1,1), stride=(1,1))
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.config.indianPines_band, self.config.conv1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.config.conv1, self.config.conv2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in, self.config.fc1),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.config.fc1, self.config.fc2),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.fc2, self.config.indianPines_class),
        )
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Extract features
        H = self.feature_extractor(x)  # N x config.conv2 x size_after_conv_and_pool_twice**2
        H = H.view(-1, self.fc_in)
        H = self.fc(H) # N x config.fc2

        # classification
        prob = self.classifier(H) # N x config.indianPines_class
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
    
    def calculate_classification_accuary(self, images, labels):
        labels = torch.squeeze(labels)
        labels = labels.type(torch.cuda.LongTensor)
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)
        print('prediction: {}, labels: {}'.format(prediction, labels))
        num_correct_classified = torch.sum(torch.eq(prediction, labels)).item()
        return num_correct_classified
    
if __name__ == '__main__':
    model = Pyramidal_ResNet(config)
    print(model)

