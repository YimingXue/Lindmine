# Import dependencies
import scipy.io
import numpy as np
from random import shuffle
import random
import spectral
import scipy.ndimage
from skimage.transform import rotate
import os
import argparse
import sys
from config import config

# IndianPines Dataset Preparation Without Augmentation

def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    percent = int(progress / float(total) * 100)
    buf = "%s|%s|  %s%d/%d %s"%(lbar_prefix, ('#' * percent).ljust(100, '-'),
        rbar_prefix, progress, total, "%d%%"%(percent))
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()

def report_progress_done():
    sys.stdout.write('\n')

if __name__ == '__main__':
    # Set random seed
    random.seed(config.indianPines_seed)

    # Load dataset
    DATA_PATH = os.path.join(os.getcwd(),"Data",config.dataset)
    input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines.mat'))['indian_pines']
    target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
    SAVE_PATH = os.path.join(DATA_PATH,config.patch_mode,"patch_size{}_seed{}".format(config.patch_size, str(config.indianPines_seed)))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Define global variables
    HEIGHT = input_mat.shape[0]
    WIDTH = input_mat.shape[1]
    BAND = config.indianPines_band
    TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
    TRAIN_PIXEL_PP,TRAIN_LABELS_PP = [],[]
    CLASSES = [] 
    COUNT = 200 # Number of pixels of each class
    OUTPUT_CLASSES = config.indianPines_class # 16
    TEST_FRAC = 0.25 # Fraction of data to be used for testing

    # Scale the input between [0,1]
    input_mat = input_mat.astype(float)
    input_mat -= np.min(input_mat)
    input_mat /= np.max(input_mat)

    # Calculate the mean of each channel for normalization
    MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=np.float32)
    for i in range(BAND):
        MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

    # Collect all available patches of each class from the given image
    for i in range(OUTPUT_CLASSES):
        CLASSES.append([])
        TRAIN_PATCH.append([]); TRAIN_LABELS.append([])
        TEST_PATCH.append([]); TEST_LABELS.append([])
        TRAIN_PIXEL_PP.append(-1); TRAIN_LABELS_PP.append(-1)
    print('Grenerating patches:')
    transpose_array = np.transpose(input_mat,(2,0,1))
    report_progress(0, HEIGHT)
    for i in range(HEIGHT):
        report_progress(i+1, HEIGHT)
        for j in range(WIDTH):
            curr_inp = transpose_array[:,i,j] # shape of curr_inp is (C,1,1)
            curr_tar = target_mat[i, j]
            if(curr_tar!=0): # Ignore patches with unknown landcover type for the central pixel
                CLASSES[curr_tar-1].append(curr_inp)
    report_progress_done()
    sys.stdout.write('\nInitial sample length per class:'.ljust(70))
    for c  in CLASSES:
        sys.stdout.write(str(len(c)).ljust(5) + ' ')
    sys.stdout.write('\n')

    # Only choose 9 classes which has more than 200 samples
    IndianPines_list = [2,3,5,6,8,10,11,12,14]
    for c in range(len(IndianPines_list)):
        IndianPines_list[c] -= 1

    # Make a train split with 200 data from each class
    for c in IndianPines_list: # for each class
        assert len(CLASSES[c]) > COUNT
        pixels_of_current_class = CLASSES[c]
        shuffle(pixels_of_current_class)
        # Make training and test splits
        TRAIN_PATCH[c] = pixels_of_current_class[:COUNT]
        # Make test splits
        TEST_PATCH[c] = pixels_of_current_class[COUNT:]
        test_split_size = len(TEST_PATCH[c])
        TEST_LABELS[c] = np.full(test_split_size, c, dtype=int)
    
    sys.stdout.write('Number of training samples per class after removal of test samples:'.ljust(70))
    for c in TRAIN_PATCH:
        sys.stdout.write(str(len(c)).ljust(5) + ' ')
    sys.stdout.write('\n')
    
    sys.stdout.write('Number of test samples per class after removal of test samples:'.ljust(70))
    for c in TEST_PATCH:
        sys.stdout.write(str(len(c)).ljust(5) + ' ')
    sys.stdout.write('\n')

    # Make pixel-pair training data
    for c in IndianPines_list: # for each class
        flag = True
        print('Class:{}'.format(c))
        report_progress(0, COUNT)
        for i in range(COUNT):
            report_progress(i+1, COUNT)
            for j in range(COUNT):
                pixel_half_one = np.expand_dims(np.array(TRAIN_PATCH[c][i]), axis=0)
                pixel_half_two = np.expand_dims(np.array(TRAIN_PATCH[c][j]), axis=0)
                pixel_mid = np.concatenate((pixel_half_one, pixel_half_two), axis=0)
                pixel = np.expand_dims(np.expand_dims(pixel_mid, axis=0), axis=0)
                if flag == True:
                    TRAIN_PIXEL_PP[c] = pixel
                    flag = False
                else:
                    TRAIN_PIXEL_PP[c] = np.concatenate((TRAIN_PIXEL_PP[c],pixel),axis=0)
        report_progress_done()
        print(TRAIN_PIXEL_PP[1].shape,c)
        exit()
                
                


# ========================================================================================================================
    # Oversample the classes which do not have at least COUNT patches 
    # in the training set and extract COUNT patches
    for i in range(OUTPUT_CLASSES):
        if(len(TRAIN_PATCH[i])<COUNT):
            tmp = TRAIN_PATCH[i]
            for j in range(COUNT//len(TRAIN_PATCH[i])):
                shuffle(TRAIN_PATCH[i])
                TRAIN_PATCH[i] = TRAIN_PATCH[i] + tmp
        shuffle(TRAIN_PATCH[i])
        TRAIN_PATCH[i] = TRAIN_PATCH[i][:COUNT]
    sys.stdout.write('Number of training samples per class after oversampling:'.ljust(70))
    for c in TRAIN_PATCH:
        sys.stdout.write(str(len(c)).ljust(4) + ' ')
    sys.stdout.write('\n')

    TRAIN_PATCH = np.asarray(TRAIN_PATCH)
    TRAIN_PATCH = TRAIN_PATCH.reshape((-1,BAND,PATCH_SIZE,PATCH_SIZE))
    TRAIN_LABELS = np.array([])
    for l in range(OUTPUT_CLASSES):
        TRAIN_LABELS = np.append(TRAIN_LABELS, np.full(COUNT, l, dtype=int))
    print('\nLength of TEST_PATCH: {}'.format(len(TEST_PATCH)))
    print('Length of TRAIN_PATCH: {}'.format(len(TRAIN_PATCH)))

    # Save the patches in segments
    # 1. Training data
    print('\nGenerating training patches: ')
    report_progress(0, len(TRAIN_PATCH)//(COUNT*2))
    for i in range(len(TRAIN_PATCH)//(COUNT*2)):
        report_progress(i+1, len(TRAIN_PATCH)//(COUNT*2))
        train_dict = {}
        start = i * (COUNT*2)
        end = (i+1) * (COUNT*2)
        file_name = 'Train_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
        train_dict["train_patch"] = TRAIN_PATCH[start:end]
        train_dict["train_labels"] = TRAIN_LABELS[start:end]
        scipy.io.savemat(os.path.join(SAVE_PATH, file_name),train_dict)
    report_progress_done()
    # 2. Test data
    print('\nGenerating test patches: ')
    report_progress(0, len(TEST_PATCH)//(COUNT*2))
    for i in range(len(TEST_PATCH)//(COUNT*2)):
        report_progress(i+1, len(TEST_PATCH)//(COUNT*2))
        test_dict = {}
        start = i * (COUNT*2)
        end = (i+1) * (COUNT*2)
        file_name = 'Test_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
        test_dict["test_patch"] = TEST_PATCH[start:end]
        test_dict["test_labels"] = TEST_LABELS[start:end]
        scipy.io.savemat(os.path.join(SAVE_PATH, file_name),test_dict)
    report_progress_done()
print('\nFinished')