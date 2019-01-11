import numpy as np
import random
import scipy.io as sio
from config import config
import sys
import os
import math

dataset = config.dataset
train_percent = config.train_percent
path = os.path.join(os.getcwd(),'Data',dataset)
mat_path = path + '/' + dataset + '_gt.mat'
gt = sio.loadmat(mat_path)
mat_name = list(dataset); mat_name[0] = mat_name[0].lower(); mat_name = ''.join(mat_name)
if dataset == 'Indian_pines_corrected':
	gt = gt['indian_pines_gt']
else:
	gt = gt[mat_name+'_gt']

nclass = np.max(gt)
print('Load dataset: {}'.format(dataset))
print('Train percent: {}'.format(train_percent))
print('Number of classes is :{}'.format(nclass))

save_path = path + '/' + str(train_percent) + '/'
if not os.path.exists(save_path):
	os.makedirs(save_path)
train_txt = open(save_path + 'train.txt', 'w')
test_txt = open(save_path + 'test.txt', 'w')

for c in range(1, nclass+1):
	y, x = np.where(gt==c)
	index = np.array([y, x]).T
	np.random.shuffle(index)
	
	nTrain = int(math.floor(y.shape[0] * train_percent))
	train = index[0:nTrain, :]
	test = index[nTrain:, :]

	for i in range(train.shape[0]):
		train_txt.write(str(train[i, 0]) + ' ' + str(train[i, 1]) + '\n')

	for i in range(test.shape[0]):
		test_txt.write(str(test[i, 0]) + ' ' + str(test[i, 1]) + '\n')

train_txt.close()
test_txt.close()