class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'crop_43' # crop_43
    inference = True # For inference
    inference_onlyTrainData = False
    maxTrain = False # Whether use limited data to train
    max_trainData = 200
    
    # train/test parameters
    model_name = 'ResNetv2' # ResNetv2/C3F4_CNN
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
    if dataset == 'crop_43':
        band = 63
        num_classes = 7
        patch_size = 27

    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    train_percent = 0.9
    val_percent = 0.0
    test_percent = 0.1
  
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