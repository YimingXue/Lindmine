import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms as T 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
from config import config
from Hyperspectral_Dataset import Hyperspectral_Dataset
import time
import datetime
import warnings
import math
import scipy.io as sio
from skimage import io
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.getcwd() + '/Model/')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

warnings.filterwarnings("ignore")
CUDA_AVAILABLE = config.cuda and torch.cuda.is_available()
torch.manual_seed(config.seed)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(config.seed)
    print('GPU is ON!')
kwargs = {'num_workers':0, 'pin_memory':True} if CUDA_AVAILABLE else {}
# EXP HEADER============================================================================
MODEL_SIGNATURE = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '-')

def train(config, kwargs):
    # IMPORT MODEL==========================================================================
    if config.model_name == 'C3F4_CNN':
        from C3F4_CNN import C3F4_CNN as Model
    elif config.model_name == 'ResNetv2':
        from ResNetv2 import ResNetv2 as Model
    else:
        raise Exception('Wrong name of the model!')
    
    # DIRECTORY FOR SAVING==================================================================
    snapshots_path = os.getcwd() + '/snapshots/patch_size' + str(config.patch_size) + '/'
    dir = snapshots_path + config.model_name + '_' + MODEL_SIGNATURE + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    path_name_current_fold = dir + config.model_name
    
    # PRINT PARAMETERS=======================================================================
    config.print_config()
    with open(path_name_current_fold + '.txt', 'a') as f:
        print('#######################PARAMETERS#######################'
          '# Dataset selection\n'
          '\tmaxTrain: {}\n'
          '\tmax_trainData: {}\n'
          '\tCBLoss_gamma: {}\n'
          
          '# train/test parameters'
          '\tmodel_name: {}\n'
          '\toptimizer: {}\n'
          '\tepochs: {}\n'
          '\tbatch_size: {}\n'
          '\tseed: {}\n'
          '\tlr: {}\n'
          '\tweight_decay: {}\n'
          
          '# data preparation parameters\n'
          '\tdataset: {}\n'
          '\tpatch_size: {}\n'
          '\tband: {}\n'
          '\tnum_classes: {}\n'
          '\ttrain_percent: {}\n'
          '\tval_percent: {}\n'
          '\ttest_percent: {}\n'.format(
          config.maxTrain, config.max_trainData, config.CBLoss_gamma, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.dataset, config.patch_size, 
          config.band, config.num_classes, config.train_percent, config.val_percent, config.test_percent),file=f)
    
    # LOAD TRAINING DATA===================================================================
    # # data_augmentation
    # transform_train = T.Compose([T.RandomHorizontalFlip(),
    #                              T.RandomVerticalFlip(),
    #                              T.ToTensor()
    #                             ])
    print('\tload training data')
    train_dataloader = DataLoader(Hyperspectral_Dataset(config,train=True), \
                                    batch_size=config.batch_size,shuffle=True,**kwargs)
    
    # CREATE AND IMPORT MODEL=============================================================
    print('\tcreate model')
    model = Model(config)
    model.apply(weights_init)
    if CUDA_AVAILABLE:
        model.cuda()
    print(model)
    with open(path_name_current_fold + '.txt', 'a') as f:
        print('#############################  MODEL  ###################################\n', file=f)
        print(model, file=f)
        print('##############################################################################\n', file=f)

    # INIT OPTIMIZER========================================================================
    print('\tinit optimizer')
    if config.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr, lr_decay=0, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.999), weight_decay=config.weight_decay)
    else:
        raise Exception('Wrong name of the optimizer!')
    # decrease lr every 40 epochs, num_epochs=40
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

    # PERFORM TRAINING EXPERIMENT==========================================================
    print('\tperform experiment\n')

    loss_batchSize = []
    for epoch in range(1, config.epochs+1):
        time_start = time.time()
        scheduler.step()

        # set model in training mode
        model.train(True)
        train_loss = 0.
        train_number = 0.

        # start training
        for batch_idx, (train_images, train_labels) in enumerate(train_dataloader):
            # data preparation
            train_images, train_labels = train_images.type(torch.FloatTensor), train_labels.type(torch.LongTensor)
            if CUDA_AVAILABLE:
                train_images, train_labels = train_images.cuda(), train_labels.cuda()
            train_images, train_labels = Variable(train_images), Variable(train_labels)
            
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = model.calculate_objective(train_images, train_labels)
            loss_batchSize.append(loss.item())
            train_loss += loss.item()
            train_number += len(train_labels)
            # backward pass
            loss.backward()
            # optimization
            optimizer.step()
        
        # calculate final loss
        train_loss = train_loss / train_number * 100
        
        time_end = time.time()
        time_elapsed = time_end - time_start
        print('Epoch %d/%d| Time: %.2fs| Loss: %.4f'%(epoch, config.epochs, time_elapsed, train_loss))
        with open(path_name_current_fold + '.txt', 'a') as f:
            print('Epoch %d/%d| Time: %.2fs| Loss: %.4f'%(epoch, config.epochs, time_elapsed, train_loss), file=f)

        if epoch % 10 == 0:
            torch.save(model, path_name_current_fold + str(epoch) +'.model')
            print('>>--{} model saved--<<'.format(path_name_current_fold+str(epoch)+'.model'))
            with open(path_name_current_fold + '.txt', 'a') as f:
                print('>>--{} model saved--<<'.format(path_name_current_fold+str(epoch)+'.model'), file=f)
            # Calculate accuary of testing dataset
            if config.dataset != 'garbage_crop_37' and config.dataset != 'img_crop_37_pool':
                test(config, kwargs, epoch, evaluate_model_assign=None, train_assign=False)
    sio.savemat(path_name_current_fold + '.mat', {'loss': loss_batchSize})

def test(config, kwargs, epoch, evaluate_model_assign=None, train_assign=False):
    # LOAD TEST DATA========================================================================
    print('\tload testing data')
    test_dataloader = DataLoader(Hyperspectral_Dataset(config,train=train_assign), \
                                    batch_size=config.batch_size,shuffle=False,**kwargs)
    
    # DIRECTORY FOR LOADING=====================================================================
    snapshots_path = os.getcwd() + '/snapshots/patch_size' + str(config.patch_size) + '/'
    dir = snapshots_path + config.model_name + '_' + MODEL_SIGNATURE + '/'
    path_name_current_fold = dir + config.model_name
    if evaluate_model_assign != None:
        evaluate_model = torch.load(snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model')
        print('>>--%s model loaded--<<'%(snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model'))
        path_name_current_fold = snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model'
        with open(path_name_current_fold + '.txt', 'a') as f:
            print('>>--%s model loaded--<<'%(snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model'), file=f)
    else:
        evaluate_model = torch.load(path_name_current_fold + str(epoch) + '.model')
        print('>>--%s model loaded--<<'%(path_name_current_fold + str(epoch) + '.model'))
        with open(path_name_current_fold + '.txt', 'a') as f:
            print('>>--%s model loaded--<<'%(path_name_current_fold + str(epoch) + '.model'), file=f)
    
    # set loss and classification accuracy to 0
    test_loss = 0.
    test_accuary = 0.
    test_number = 0.
    accuracy_per_class = np.zeros(config.num_classes)
    number_per_class = np.zeros(config.num_classes)

    # CALCULATE CLASSIFICATION RESULT AND LOSS FOR TEST SET====================================
    # set evaluate_model to evaluation mode
    evaluate_model.eval()

    t_ll_s = time.time()
    for batch_idx, (test_images, test_labels) in enumerate(test_dataloader):
        # data preparation
        test_images, test_labels = test_images.type(torch.FloatTensor), test_labels.type(torch.LongTensor)
        if CUDA_AVAILABLE:
            test_images, test_labels = test_images.cuda(), test_labels.cuda()
        test_images, test_labels = Variable(test_images), Variable(test_labels)

        # calculate loss and result
        loss = evaluate_model.calculate_objective(test_images, test_labels).item()
        accuary, accuracy_per_class, number_per_class = evaluate_model.calculate_classification_accuary(test_images,test_labels,
                                                                                            accuracy_per_class,number_per_class)
        test_loss += loss
        test_accuary += accuary
        accuary = accuary / len(test_labels) * 100
        test_number += len(test_labels)
        if batch_idx % 50 == 0:
            print('\tBatch_idx: %d | Loss: %.4f | Accuracy: %d'%(batch_idx, loss, accuary))
            with open(path_name_current_fold + '.txt', 'a') as f:
                print('\tBatch_idx: %d | Loss: %.4f | Accuracy: %d'%(batch_idx, loss, accuary), file=f)
    
    t_ll_e = time.time()
    # calculate final loss and accuary
    test_loss = test_loss / test_number * 100
    test_accuary = test_accuary / test_number * 100

    Pe = 0.
    AA = 0.
    exist_classes = 0
    for i in range(len(accuracy_per_class)):
        if number_per_class[i] != 0:
            Pe += accuracy_per_class[i] * number_per_class[i]

            exist_classes += 1
            accuracy_pc = accuracy_per_class[i]/number_per_class[i]*100
            AA += accuracy_pc
            print('  Class %d, accuracy: %.2f'%(i, accuracy_pc))
            with open(path_name_current_fold + '.txt', 'a') as f:
                print('  Class %d, accuracy: %.2f'%(i, accuracy_pc),file=f)
    P0 = test_accuary / 100
    Pe = Pe / (test_number**2)
    Kappa = (P0 - Pe) / (1 - Pe) * 100
    AA /= exist_classes
    print('Testing Loss: %.4f | OA: %.2f | AA: %.2f | Kappa: %.2f | Time: %.2f'%(test_loss, test_accuary, AA, Kappa, t_ll_e-t_ll_s))
    with open(path_name_current_fold + '.txt', 'a') as f:
        print('Testing Loss: %.4f | OA: %.2f | AA: %.2f | Kappa: %.2f | Time: %.2f'%(test_loss, test_accuary, AA, Kappa, t_ll_e-t_ll_s), file=f)

def inference(config, kwargs, epoch, evaluate_model_assign=None, train_assign=False):
    # LOAD INFERENCE DATA========================================================================
    print('\tload inference data')
    inference_dataloader = DataLoader(Hyperspectral_Dataset(config,train=train_assign), \
                                    batch_size=500,shuffle=False,**kwargs)
    
    # DIRECTORY FOR LOADING=====================================================================
    snapshots_path = os.getcwd() + '/snapshots/patch_size' + str(config.patch_size) + '/'
    dir = snapshots_path + config.model_name + '_' + MODEL_SIGNATURE + '/'
    path_name_current_fold = dir + config.model_name
    if evaluate_model_assign != None:
        evaluate_model = torch.load(snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model')
        print('>>--%s model loaded--<<'%(snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model'))
        path_name_current_fold = snapshots_path + evaluate_model_assign + '/' + config.model_name + str(epoch) + '.model'
    else:
        evaluate_model = torch.load(path_name_current_fold + str(epoch) + '.model')
        print('>>--%s model loaded--<<'%(path_name_current_fold + str(epoch) + '.model'))

    # CALCULATE CLASSIFICATION RESULT AND LOSS FOR INFERENCE SET====================================
    # set evaluate_model to evaluation mode
    evaluate_model.eval()

    t_ll_s = time.time()
    path = os.path.join(os.getcwd(),'Data',config.dataset)
    mat_path = path + '/' + config.dataset + '.mat'
    mat_data = sio.loadmat(mat_path)
    mat_name = list(config.dataset); mat_name[0] = mat_name[0].lower(); mat_name = ''.join(mat_name)
    gt = mat_data[mat_name]
    Height, Width = gt.shape[0], gt.shape[1]
    image = np.zeros([Height, Width, 3], dtype=np.uint8)
    tp = 0; tn = 0; fp = 0; fn = 0
    for h in range(Height):
        for w in range(Width):
            image[h,w,0:3] = (255,255,255)
    
    for batch_idx, (inference_images, inference_labels, h, w) in enumerate(inference_dataloader):
        # data preparation
        inference_images = inference_images.type(torch.FloatTensor)
        inference_labels = inference_labels.type(torch.FloatTensor)
        if CUDA_AVAILABLE:
            inference_images = inference_images.cuda()
            inference_labels = inference_labels.cuda()
        inference_images = Variable(inference_images)
        inference_labels = Variable(inference_labels)

        # calculate loss and result
        prediction = evaluate_model.inference_classification(inference_images)
        for i in range(len(prediction)):
            class_number = 7 - 1
            if prediction[i] == class_number and inference_labels[i] == class_number:
                tp += 1
            if prediction[i] == class_number and inference_labels[i] != class_number:
                fp += 1
            if prediction[i] != class_number and inference_labels[i] == class_number:
                fn += 1
            if prediction[i] != class_number and inference_labels[i] != class_number:
                tn += 1

            if prediction[i] == class_number:
                image[h[i],w[i],0:3] = (0,0,255)
            else:
                image[h[i],w[i],0:3] = (255,255,255)

            # if prediction[i] == 1:
            #     image[h[i],w[i],0:3] = (0,255,0)
            # elif prediction[i] == 2:
            #     image[h[i],w[i],0:3] = (0,128,0)
            # elif prediction[i] == 3:
            #     image[h[i],w[i],0:3] = (255,255,0)
            # elif prediction[i] == 4:
            #     image[h[i],w[i],0:3] = (255,174,200)
            # elif prediction[i] == 5:
            #     image[h[i],w[i],0:3] = (128,0,0)
            # elif prediction[i] == 6:
            #     image[h[i],w[i],0:3] = (0,128,192)
            # elif prediction[i] == 7:
            #     image[h[i],w[i],0:3] = (0,0,255)
            # elif prediction[i] == 8:
            #     image[h[i],w[i],0:3] = (253,236,166)
            # elif prediction[i] == 9:
            #     image[h[i],w[i],0:3] = (255,202,24)
            # elif prediction[i] == 10:
            #     image[h[i],w[i],0:3] = (255,127,39)
            # elif prediction[i] == 11:
            #     image[h[i],w[i],0:3] = (185,122,86)
            # elif prediction[i] == 12:
            #     image[h[i],w[i],0:3] = (255,0,0)
            # elif prediction[i] == 13:
            #     image[h[i],w[i],0:3] = (255,128,128)
            # elif prediction[i] == 14:
            #     image[h[i],w[i],0:3] = (200,100,100)
            # elif prediction[i] == 15:
            #     image[h[i],w[i],0:3] = (72,30,7)
            # elif prediction[i] == 16:
            #     image[h[i],w[i],0:3] = (0,168,243)
            # elif prediction[i] == 17:
            #     image[h[i],w[i],0:3] = (63,72,204)
            # elif prediction[i] == 18:
            #     image[h[i],w[i],0:3] = (196,255,14)
            # elif prediction[i] == 19:
            #     image[h[i],w[i],0:3] = (74,85,38)
            # else:
            #     image[h[i],w[i],0:3] = (255,255,255)
        if batch_idx % 1 == 0:
            print('batch_idx {}'.format(batch_idx))
    
    t_ll_e = time.time()
    print('Using %.2f seconds'%(t_ll_e-t_ll_s))
    
    save_path = os.path.join(os.getcwd(),'Data',config.dataset)
    image_save_path = save_path + '/' + config.dataset + '_inference_PS' + str(config.patch_size) + '_epoch' + str(epoch) + '.eps'
    # Using plt save GT figure
    plt.figure(config.dataset)
    plt.imshow(image)
    figure = plt.gcf() # gtf:Get Current Figure
    figure.savefig(image_save_path, format='eps', dpi=600)
    plt.show()
    print('Save Inference image')
    print('tp: {}, fp: {}, tn: {}, fn: {}'.format(tp, fp, tn, fn))
    print('precision: {}, recall: {}'.format(tp/(tp+fp), tp/(tp+fn)))


if __name__ == '__main__':
    if config.inference == False:
        train(config, kwargs)
        # epoch = 10
        # evaluate_model_assign = 'C3F4_CNN_RON_2019-01-17_15-32-27'
        # test(config, kwargs, epoch, evaluate_model_assign)
    else:
        epoch = 80
        evaluate_model_assign = 'ResNetv2_2019-05-17_10-51-15_p0.8'
        inference(config, kwargs, epoch, evaluate_model_assign)
