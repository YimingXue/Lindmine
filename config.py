class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'garbage_crop_37' # Indian_pines/Indian_pines_corrected/PaviaU/Pavia/crop_43/garbage_crop_37
    
    # train/test parameters
    model_name = 'C3F4_CNN' # Pyramidal_ResNet/SimpleFC/C3F4_CNN
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 150
    step_size = 40
    batch_size = 100
    seed = 75
    lr = 0.01 # 0.1
    weight_decay = 1e-4 # 1e-4

    # Dataset
    if dataset == 'Indian_pines':
        band = 220
        num_classes = 16
    if dataset == 'Indian_pines_corrected':
        band = 200
        num_classes = 16
    if dataset == 'PaviaU':
        band = 103
        num_classes = 9
    if dataset == 'crop_43':
        band = 63
        num_classes = 15
    if dataset == 'garbage_crop_37':
        band = 63
        num_classes = 2
    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    patch_size = 31
    train_percent = 0.2
    val_percent = 0.0
    test_percent = 0.8
    inference = True

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