import numpy as np
import random
import scipy.io as sio
import sys
sys.path.insert(0,'/home/xueyiming/Landmine/')
from config import config
import os
import math

# split the training and test dataset with exact percent
if __name__ == '__main__':
	dataset = config.dataset
	train_percent = config.train_percent
	val_percent = config.val_percent
	test_percent = config.test_percent
	path = os.path.join(os.getcwd(),'Data',dataset)
	mat_path = path + '/' + dataset + '_gt.mat'
	gt = sio.loadmat(mat_path)
	mat_name = list(dataset); mat_name[0] = mat_name[0].lower(); mat_name = ''.join(mat_name)
	gt = gt[mat_name+'_gt']

	print('Load dataset: {}'.format(dataset))
	print('Number of classes is :{}'.format(config.num_classes))

	Height, Width = gt.shape[0], gt.shape[1]

	save_path = path + '/Inference/'
	if not os.path.exists(save_path):
	    os.makedirs(save_path)
	inference_txt = open(save_path + 'inference.txt', 'w')

	for h in range(Height):
	    for w in range(Width):
		    inference_txt.write(config.dataset + ' ' + str(h) + ' ' + str(w) + '\n')

	inference_txt.close()
