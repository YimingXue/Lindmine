from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
# from dataset.aug import *
import random 
import numpy as np 
import os 
import torch
import scipy.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# rotate 90,180,270
def rotation(image):
    channel = image.shape[0]
    n = random.randint(1,3)
    length = image.shape[1]
    for _ in range(n):
        for c in range(channel):
            for i in range(length):
                for j in range(i+1,length):
                    temp = image[c][i][j]
                    image[c][i][j] = image[c][j][i]
                    image[c][j][i] = temp
            for i in range(len(image[c])):
                image[c][i] = image[c][i][::-1]
    return image

# define dataset
class IndianPinesDataset(Dataset):
    def __init__(self,patch_size=config.patch_size,train=True):
        self.patch_size = patch_size
        self.train = train
        self.train_images = None; self.train_labels = None
        self.test_images = None; self.test_labels = None
        
        data_path = os.path.join(os.getcwd(),'Data',config.dataset,config.patch_mode,'patch_size{}_seed{}'.format(self.patch_size, str(config.indianPines_seed)))
        file_list = os.listdir(data_path)
        
        first_flag = True
        for file in file_list:
            if self.train == True and file.split('_')[0] == 'Train':
                if first_flag == True:
                    image = sio.loadmat(os.path.join(data_path, file))['train_patch'] # [400,220,patch_size,patch_size]
                    self.train_images = image
                    label = sio.loadmat(os.path.join(data_path, file))['train_labels'] # [1,400]
                    self.train_labels = np.transpose(label)
                    first_flag = False
                elif first_flag == False:
                    image = sio.loadmat(os.path.join(data_path, file))['train_patch']
                    self.train_images = np.concatenate((self.train_images, image))
                    label = sio.loadmat(os.path.join(data_path, file))['train_labels']
                    self.train_labels = np.concatenate((self.train_labels, np.transpose(label)))
            elif self.train == False and file.split('_')[0] == 'Test':
                if first_flag == True:
                    image = sio.loadmat(os.path.join(data_path, file))['test_patch'] # [400, 220, patch_size, patch_size]
                    self.test_images = image
                    label = sio.loadmat(os.path.join(data_path, file))['test_labels'] # [1,400]
                    self.test_labels = np.transpose(label)
                    first_flag = False
                elif first_flag == False:
                    image = sio.loadmat(os.path.join(data_path, file))['test_patch']
                    self.test_images = np.concatenate((self.test_images, image))
                    label = sio.loadmat(os.path.join(data_path, file))['test_labels']
                    self.test_labels = np.concatenate((self.test_labels, np.transpose(label)))
        
        if self.train == True:
            print('\tLength of training samples is {}'.format(self.train_images.shape[0]))
        else:
            print('\tLength of testing samples is {}'.format(self.test_images.shape[0]))

    def __getitem__(self,index):
        if self.train == True:
            # num = random.randint(0,1)
            # if num == 0 :
            #     self.train_images[index] = self.train_images[index][:,::-1,:] # Flip patch up-down
            # if num == 1 :
            #     self.train_images[index] = self.train_images[index][:,:,::-1] # Flip patch left-right
            # if num == 2 :
            #     self.train_images[index] = rotation(self.train_images[index]) # Rotate patch for 90/180/270
            return self.train_images[index], self.train_labels[index]
        else:
            return self.test_images[index], self.test_labels[index]

    def __len__(self):
        if self.train == True:
            return self.train_images.shape[0]
        else:
            return self.test_images.shape[0]

    
if __name__ == '__main__':
    train_dataloader = DataLoader(IndianPinesDataset(patch_size=config.patch_size,train=True), \
                                    batch_size=config.batch_size,shuffle=True)
    test_dataloader = DataLoader(IndianPinesDataset(patch_size=config.patch_size,train=False), \
                                    batch_size=config.batch_size,shuffle=False)
    for iter,(train_images,train_labels) in enumerate(train_dataloader):
        print(iter, train_images.shape, train_labels.shape)
    for iter,(test_images,test_labels) in enumerate(test_dataloader):
        print(iter, test_images.shape, test_labels.shape)