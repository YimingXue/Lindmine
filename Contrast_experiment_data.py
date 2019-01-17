import numpy as np
import random
import scipy.io as sio
from config import config
import sys
import os
import math

if __name__ == '__main__':
	dataset = config.dataset
	train_percent = config.train_percent
	max_trainData = config.max_trainData
	val_percent = config.val_percent
	test_percent = config.test_percent
	path = os.path.join(os.getcwd(),'Data',dataset)
	mat_path = path + '/' + dataset + '_gt.mat'
	gt = sio.loadmat(mat_path)
	mat_name = list(dataset); mat_name[0] = mat_name[0].lower(); mat_name = ''.join(mat_name)
	if dataset == 'Indian_pines_corrected':
		gt = gt['indian_pines_gt']
	else:
		gt = gt[mat_name+'_gt']
	
	print('Load dataset: {}'.format(dataset))
	print('Max train number: {}'.format(max_trainData))
	print('Val percent: {}'.format(val_percent))
	print('Test percent: {}'.format(test_percent))
	print('Number of classes is :{}'.format(config.num_classes))

	save_path = path + '/' + str(max_trainData) + '/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	train_txt = open(save_path + 'train.txt', 'w')
	test_txt = open(save_path + 'test.txt', 'w')
	val_txt = open(save_path + 'val.txt', 'w')

	classList = []
	for i in range(1, config.num_classes+1):
		if i in gt:
			classList.append(i)
	
	for c in classList:
		y, x = np.where(gt==c)
		index = np.array([y, x]).T
		np.random.shuffle(index)
		
		nTrain = max_trainData
		nVal = int(math.floor(y.shape[0] * val_percent))
		nTest = int(math.floor(y.shape[0] * test_percent))
		train = index[0:nTrain, :]
		val = index[nTrain:nTrain+nVal, :]
		test = index[nTrain+nVal:nTrain+nVal+nTest, :]

		for i in range(train.shape[0]):
			train_txt.write(config.dataset + ' ' + str(train[i, 0]) + ' ' + str(train[i, 1]) + '\n')

		for i in range(val.shape[0]):
			val_txt.write(config.dataset + ' ' + str(val[i, 0]) + ' ' + str(val[i, 1]) + '\n')

		for i in range(test.shape[0]):
			test_txt.write(config.dataset + ' ' + str(test[i, 0]) + ' ' + str(test[i, 1]) + '\n')

	train_txt.close()
	test_txt.close()
	val_txt.close()

