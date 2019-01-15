import numpy as np
import random
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(description='select training images')
parser.add_argument('--Rate', type=int, default=20, help='rate of train image')
parser.add_argument('--nClass', type=int, default=9, help='rate of train image')

opt = parser.parse_args()
print(opt)

imgList = ['crop_43']
rate = opt.Rate / 100.0
file1 = open('./train_' + str(opt.Rate) + '.txt', 'w')
file2 = open('./test_' + str(opt.Rate) + '.txt', 'w')

print('##### Select Train #####')
print('#####     Over     #####')
for name in imgList:
	print(name)
	gt = sio.loadmat('./'+ name + '_gt.mat')
	gt = gt[name + '_gt']

	classList = []
	for i in range(1, opt.nClass+1):
		if i in gt:
			classList.append(i)

	for c in classList:
		y, x = np.where(gt==c)
		index = np.array([y, x]).T
		np.random.shuffle(index)
		
		nTrain = int(round(y.shape[0] * rate))
		train = index[0:nTrain, :]
		test = index[nTrain:, :]

		for i in range(train.shape[0]):
			file1.write(name + ' ' + str(train[i, 0]) + ' ' + str(train[i, 1]) + '\n')

		for i in range(test.shape[0]):
			file2.write(name + ' ' + str(test[i, 0]) + ' ' + str(test[i, 1]) + '\n')

file1.close()
file2.close()
print('#####     Over     #####')
