import torch
import torch.nn as nn
from config import config
import math

class SimpleNet(nn.Module):
    def __init__(self, config):
        super(SimpleNet, self).__init__()
        self.config = config

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1,9), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2,1), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,1)),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(1,3), stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,1)),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1,3), stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1040, out_features=80),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(in_features=80, out_features=80),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(80, self.config.indianPines_class),
        )
        self.Softmax = nn.Softmax(dim=0)  
    
    def forward(self, x):
        # Extract features
        H = self.feature_extractor(x) 
        H = H.view(-1, 1040)
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
    model = SimpleNet(config)
    print(model)

