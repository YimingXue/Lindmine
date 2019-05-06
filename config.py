class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'PaviaU' # Indian_pines/Indian_pines_corrected/PaviaU/
                           # PaviaU/PaviaU_1D/PaviaU_2D/PaviaU_3D
                           # crop_43/crop_43_1D/crop_43_2D/crop_43_3D
                           # crop_59/crop_59_1D/crop_59_2D/crop_59_3D
    inference = False # For garbage_crop_37 inference
    inference_onlyTrainData = False 
    maxTrain = True # Whether use limited data to train
    max_trainData = 200
    
    # train/test parameters
    model_name = 'ResNetv3_CBLoss' # ResNetv1_CBLoss/ResNetv2_CBLoss/ResNetv3_CBLoss
                                   # ResNetv2/SimpleFC/C3F4_CNN
    CBLoss_gamma = 1
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 80
    step_size = 20
    batch_size = 100
    seed = 80 
    lr = 0.01
    weight_decay = 1e-4

    # Dataset
    if dataset == 'Indian_pines':
        band = 220
        num_classes = 16
        patch_size = 29
    if dataset == 'PaviaU':
        band = 103
        num_classes = 9
        patch_size = 27

    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    train_percent = 0.75
    val_percent = 0.0
    test_percent = 0.25

    # SimpleFC parameters
    FC_1 = 500
    FC_2 = 350
    FC_3 = 150
    FC_4 = 16
  
    def print_config(self):
        print('#######################PARAMETERS#######################'
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
          '\tdataset: {}\n'
          '\tpatch_size: {}\n'
          '\tband: {}\n'
          '\tnum_classes: {}\n'
          '\ttrain_percent: {}\n'
          '\tval_percent: {}\n'.format(
          config.cuda, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.dataset, config.patch_size, 
          config.band, config.num_classes, config.train_percent, config.val_percent))
        print('#'*60)
        print('\n')

config = DefaultConfigs()
if __name__ == '__main__':
    config.print_config()