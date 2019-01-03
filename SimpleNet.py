import torch
import torch.nn as nn
from config import config
import math
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision

class SimpleNet(nn.Module):
    def __init__(self, config):
        super(SimpleNet, self).__init__()
        self.config = config
        self.size_after_conv_and_pool_twice = int(math.floor((math.floor((self.config.patch_size-5+1)/2.0)-5+1)/2.0))
        self.fc_in = self.config.conv2*self.size_after_conv_and_pool_twice**2

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.config.indianPines_band, self.config.conv1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.config.conv1, self.config.conv2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        torch.nn.init.xavier_uniform_(self.feature_extractor[0].weight)
        self.feature_extractor[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.feature_extractor[3].weight)
        self.feature_extractor[3].bias.data.zero_()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in, self.config.fc1),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.config.fc1, self.config.fc2),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc[2].weight)
        self.fc[2].bias.data.zero_()

        self.classifier = nn.Sequential(
            nn.Linear(self.config.fc2, self.config.indianPines_class),
        )
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.zero_()

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
    model = SimpleNet(config)
    print(model)

