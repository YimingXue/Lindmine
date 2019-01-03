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

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# define dataset
class IndianPinesDataset(Dataset):
    def __init__(self,patch_size=config.patch_size,train=True):
        self.patch_size = patch_size
        self.train = train
        self.train_images = None; self.train_labels = None
        self.test_images = None; self.test_labels = None
        
        data_path = os.path.join(os.getcwd(),'Data',config.patch_mode,'patch_size{}_seed{}'.format(self.patch_size, str(config.indianPines_seed)))
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