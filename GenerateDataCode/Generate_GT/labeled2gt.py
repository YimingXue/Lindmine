import os
import numpy as np
import argparse
from skimage import io
import scipy.io as sio
import operator
import sys
import time
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Convert Labels to Groud Truth')
parser.add_argument('--lpp', type=str, default='garbage_crop_37.png', help='Labeled Patch Path')

args = parser.parse_args()

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
    patch_labeled = io.imread(args.lpp)
    height, width, channel = patch_labeled.shape
    print(height, width)
    exit()

    # Prepare empty files for GT pictureFile and matFile
    patch_gt_pic = np.zeros([height, width, 3], dtype=np.uint8)
    patch_gt_mat = np.zeros([height, width], dtype=np.uint8)
    print('Rate of progress:')
    report_progress(0, height)
    for h in range(height):
        report_progress(h+1, height)
        for w in range(width):
            if operator.eq(patch_labeled[h, w, 0:3].tolist(), [200,100,200]) == True: # garbage 1
                patch_gt_pic[h, w, :] = [78,29,76]
                patch_gt_mat[h, w] = 1
            if operator.eq(patch_labeled[h, w, 0:3].tolist(), [78,29,76]) == True: # garbage 2
                patch_gt_pic[h, w, :] = [78,29,76]
                patch_gt_mat[h, w] = 1
            if operator.eq(patch_labeled[h, w, 0:3].tolist(), [100,170,200]) == True: # none garbage
                patch_gt_pic[h, w, :] = [100,170,200]
                patch_gt_mat[h, w] = 2
    report_progress_done()
    
    # Save GT pictureFile 
    filePath, fileName = os.path.split(args.lpp)
    name, postfix = os.path.splitext(fileName)
    picFileName = name + '_gt' + postfix
    saveGTpath = os.path.join(filePath, picFileName)
    io.imsave(saveGTpath, patch_gt_pic)
    # Save GT matFile
    matFileName = name + '_gt.mat'
    saveMatpath = os.path.join(filePath, matFileName)
    sio.savemat(saveMatpath, {name+'_gt': patch_gt_mat})

    print('Successfully generated!')
    