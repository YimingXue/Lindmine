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
# 0.25 for training and 0.75 for testing

def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    percent = int(progress / float(total) * 100)
    buf = "%s|%s|  %s%d/%d %s"%(lbar_prefix, ('#' * percent).ljust(100, '-'),
        rbar_prefix, progress, total, "%d%%"%(percent))
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()

def report_progress_done():
    sys.stdout.write('\n')

def Patch_TopLeft(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which 
    is at (height_index, width_index)
    
    Inputs: 
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch
    
    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
    whose top left corner is at (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat,(2,0,1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

def Patch_Center(height_index,width_index):
    """
    Returns a mean-normalized patch, the center corner of which 
    is at (height_index, width_index)
    
    Inputs: 
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch
    
    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
    whose top left corner is at (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat,(2,0,1))
    patch = np.zeros((BAND, PATCH_SIZE, PATCH_SIZE))
    offset = (PATCH_SIZE-1)//2
    h_index = 0; w_index = 0
    for h in range(height_index-offset, height_index+offset+1):
        for w in range(width_index-offset, width_index+offset+1):
            if h<0 or h>=HEIGHT or w<0 or w>=WIDTH:
                continue
            else:
                patch[:,h-height_index+offset,w-width_index+offset] = transpose_array[:,h,w]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    return np.array(mean_normalized_patch)

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
    PATCH_SIZE = config.patch_size
    HEIGHT = input_mat.shape[0]
    WIDTH = input_mat.shape[1]
    BAND = config.indianPines_band
    TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
    CLASSES = [] 
    COUNT = 200 # Number of patches of each class
    OUTPUT_CLASSES = config.indianPines_class
    TEST_FRAC = 0.25 # Fraction of data to be used for testing
    print('patch_size: {}\n'.format(PATCH_SIZE))

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
    print('Grenerating patches:')
    if config.patch_mode == 'TopLeft':
        report_progress(0, HEIGHT - PATCH_SIZE + 1)
        for i in range(HEIGHT - PATCH_SIZE + 1):
            report_progress(i+1, HEIGHT - PATCH_SIZE + 1)
            for j in range(WIDTH - PATCH_SIZE + 1):
                curr_inp = Patch_TopLeft(i,j) # shape of curr_inp is (C,H,W)
                curr_tar = target_mat[i, j]
                if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
                    CLASSES[curr_tar-1].append(curr_inp)
    if config.patch_mode == 'Center':
        report_progress(0, HEIGHT)
        for i in range(HEIGHT):
            report_progress(i+1, HEIGHT)
            for j in range(WIDTH):
                curr_inp = Patch_Center(i,j) # shape of curr_inp is (C,H,W)
                curr_tar = target_mat[i, j]
                if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
                    CLASSES[curr_tar-1].append(curr_inp)
    report_progress_done()
    sys.stdout.write('\nInitial sample length per class:'.ljust(70))
    for c  in CLASSES:
        sys.stdout.write(str(len(c)).ljust(4) + ' ')
    sys.stdout.write('\n')

    # Make a test split with 25% data from each class
    for c in range(OUTPUT_CLASSES): #for each class
        class_population = len(CLASSES[c])
        test_split_size = int(class_population*TEST_FRAC)
            
        patches_of_current_class = CLASSES[c]
        shuffle(patches_of_current_class)
        
        # Make training and test splits
        TRAIN_PATCH.append(patches_of_current_class[:-test_split_size])
            
        TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
        TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))
    sys.stdout.write('Number of training samples per class after removal of test samples:'.ljust(70))
    for c in TRAIN_PATCH:
        sys.stdout.write(str(len(c)).ljust(4) + ' ')
    sys.stdout.write('\n')

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


