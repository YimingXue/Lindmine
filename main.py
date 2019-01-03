import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
from config import config
from IndianPinesDataset import IndianPinesDataset
import time
import datetime
import warnings

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)
    # elif isinstance(m, nn.BatchNorm2d):
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)

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
    if config.model_name == 'SimpleNet':
        from SimpleNet import SimpleNet as Model
    elif config.model_name == 'SimpleFC':
        from SimpleFC import SimpleFC as Model
    else:
        raise Exception('Wrong name of the model!')
    
    # DIRECTORY FOR SAVING==================================================================
    snapshots_path = os.getcwd() + '/snapshots/patch_size' + str(config.patch_size) + '/'
    dir = snapshots_path + config.model_name + '_' + MODEL_SIGNATURE + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    path_name_current_fold = dir + config.model_name
    
    # PRINT PARAMETERS
    config.print_config()
    with open(path_name_current_fold + '.txt', 'a') as f:
        print('#############################  PARAMETERS  ###################################\n'
          '# cuda\n'
          '\tcuda: {}\n'
          '# train/test parameters'
          '\tmodel_name: {}\n'
          '\toptimizer: {}\n'
          '\tepochs: {}\n'
          '\tbatch_size: {}\n'
          '\tseed: {}\n'
          '\tlr: {}\n'
          '\tweight_decay: {}\n'
          '# data preparation parameters\n'
          '\tpatch_size: {}\n'
          '\tindianPines_band: {}\n'
          '\tindianPines_class: {}\n'
          '# SimpleNet parameters\n'
          '\tconv1: {}\n'
          '\tconv2: {}\n'
          '\tfc1: {}\n'
          '\tfc2: {}\n'
          '# SimpleFC parameters\n'
          '\tFC_1: {}\n'
          '\tFC_2: {}\n'
          '\tFC_3: {}\n'
          '\tFC_4: {}\n'
          '##############################################################################\n'
          .format(
          config.cuda, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.patch_size, 
          config.indianPines_band, config.indianPines_class,
          config.conv1, config.conv2,config.fc1, config.fc2,
          config.FC_1, config.FC_2, config.FC_3, config.FC_4
          ), file=f)
    
    # LOAD TRAINING DATA===================================================================
    print('\tload training data')
    train_dataloader = DataLoader(IndianPinesDataset(patch_size=config.patch_size,train=True), \
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

    for epoch in range(1, config.epochs+1):
        time_start = time.time()
        scheduler.step()

        # set model in training mode
        model.train(True)
        train_loss = 0.

        # start training
        for batch_idx, (train_images, train_labels) in enumerate(train_dataloader):
            # data preparation
            train_images, train_labels = train_images.type(torch.FloatTensor), train_labels.type(torch.FloatTensor)
            if CUDA_AVAILABLE:
                train_images, train_labels = train_images.cuda(), train_labels.cuda()
            train_images, train_labels = Variable(train_images), Variable(train_labels)

            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = model.calculate_objective(train_images, train_labels)
            train_loss += loss.item()
            # backward pass
            loss.backward()
            # optimization
            optimizer.step()
        
        # calculate final loss
        train_loss /= len(train_dataloader)
        
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
            # Calculate accuary of training dataset
            test(config, kwargs, epoch, evaluate_model_assign=None, train_assign=False)

def test(config, kwargs, epoch, evaluate_model_assign=None, train_assign=False):
    # LOAD TRAINING DATA========================================================================
    print('\tload testing data')
    test_dataloader = DataLoader(IndianPinesDataset(patch_size=config.patch_size,train=train_assign), \
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
    
    # set loss and classification error to 0
    test_loss = 0.
    test_accuary = 0.

    # CALCULATE CLASSIFICATION RESULT AND LOSS FOR TEST SET====================================
    # set evaluate_model to evaluation mode
    evaluate_model.eval()

    t_ll_s = time.time()
    for batch_idx, (test_images, test_labels) in enumerate(test_dataloader):
        # data preparation
        test_images, test_labels = test_images.type(torch.FloatTensor), test_labels.type(torch.FloatTensor)
        if CUDA_AVAILABLE:
            test_images, test_labels = test_images.cuda(), test_labels.cuda()
        test_images, test_labels = Variable(test_images), Variable(test_labels)

        # calculate loss and result
        loss = evaluate_model.calculate_objective(test_images, test_labels).item()
        accuary = evaluate_model.calculate_classification_accuary(test_images, test_labels)
        test_loss += loss
        test_accuary += accuary
        print('\tBatch_idx: %d | Loss: %.4f | AccuracyNumber: %d'%(batch_idx, loss, accuary))
        with open(path_name_current_fold + '.txt', 'a') as f:
            print('\tBatch_idx: %d | Loss: %.4f | AccuracyNumber: %d'%(batch_idx, loss, accuary), file=f)
    
    t_ll_e = time.time()
    # calculate final loss and accuary
    test_loss /= len(test_dataloader)
    test_accuary /= len(test_dataloader)

    print('Testing Loss: %.4f | Testing Accuracy: %.4f | Time: %.2f'%(test_loss, test_accuary, t_ll_e-t_ll_s))
    with open(path_name_current_fold + '.txt', 'a') as f:
        print('Testing Loss: %.4f | Testing Accuracy: %.4f | Time: %.2f'%(test_loss, test_accuary, t_ll_e-t_ll_s), file=f)

if __name__ == '__main__':
    train(config, kwargs)
    # epoch = 40
    # evaluate_model_assign = 'SimpleFC_2019-01-03_13-08-17'
    # test(config, kwargs, epoch, evaluate_model_assign)
