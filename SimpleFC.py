import torch
import torch.nn as nn
from config import config
import math
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision

class SimpleFC(nn.Module):
    def __init__(self, config):
        super(SimpleFC, self).__init__()
        self.config = config
        self.fc_in = self.config.indianPines_band * self.config.patch_size**2

        self.FC = nn.Sequential(
            nn.Linear(self.fc_in, self.config.FC_1),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.config.FC_1, self.config.FC_2),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.config.FC_2, self.config.FC_3),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.config.FC_3, self.config.FC_4),
            # nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, self.fc_in)
        # Extract features
        prob = self.FC(x)  # N x config.conv2 x size_after_conv_and_pool_twice**2
        # classification
        softmax_prob = self.Softmax(prob)
        return prob, softmax_prob
    
    def calculate_objective(self, images, labels):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()
        output = loss(prob,labels)
        return output
    
    def calculate_classification_accuary(self, images, labels):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)
        # print('Prediction: {}, Labels: {}'.format(prediction, labels))
        num_correct_classified = torch.sum(torch.eq(prediction, labels)).item()
        return num_correct_classified
    
if __name__ == '__main__':
    model = SimpleFC(config)
    print(model)

