class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'crop_43' # Indian_pines/Indian_pines_corrected/PaviaU/Pavia/crop_43
    
    # train/test parameters
    model_name = 'C3F4_CNN' # Pyramidal_ResNet/SimpleFC/C3F4_CNN
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 100
    step_size = 40
    batch_size = 100
    seed = 75
    lr = 0.01 # 0.1
    weight_decay = 1e-4 # 1e-4

    # IndianPines data preparation parameters
    if dataset == 'Indian_pines':
        indianPines_band = 220
    elif dataset == 'Indian_pines_corrected':
        indianPines_band = 200
    indianPines_class = 16

    # PaviaU data preparation parameters
    PaviaU_band = 103
    PaviaU_class = 9

    # crop_43 data preparation parameters
    crop_43_band = 63
    crop_43_class = 15

    # Dataset
    if dataset == 'Indian_pines' or dataset == 'Indian_pines_corrected':
        band = indianPines_band
        num_classes = indianPines_class
    if dataset == 'PaviaU':
        band = PaviaU_band
        num_classes = PaviaU_class
    if dataset == 'crop_43':
        band = crop_43_band
        num_classes = crop_43_class
    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    patch_size = 21
    train_percent = 0.01
    val_percent = 0.0
    test_percent = 0.1

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