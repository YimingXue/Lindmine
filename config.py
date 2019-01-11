class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'Indian_pines' # Indian_pines/Indian_pines_corrected/PaviaU/Pavia
    train_percent = 0.2

    # train/test parameters
    model_name = 'Pyramidal_ResNet' # Pyramidal_ResNet/SimpleFC
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 150
    step_size = 40
    batch_size = 100
    seed = 75
    lr = 0.001 # 0.1
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

    # Dataset
    if dataset == 'Indian_pines' or dataset == 'Indian_pines_corrected':
        band = indianPines_band
        num_classes = indianPines_class
    if dataset == 'PaviaU':
        band = PaviaU_band
        num_classes = PaviaU_class
    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    patch_size = 11

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
          
          '# SimpleFC parameters\n'
          '\tFC_1: {}\n'
          '\tFC_2: {}\n'
          '\tFC_3: {}\n'
          '\tFC_4: {}\n'.format(
          config.cuda, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.dataset, config.patch_size, 
          config.band, config.num_classes,
          config.FC_1, config.FC_2, config.FC_3, config.FC_4
          ))
        print('#'*60)
        print('\n')

config = DefaultConfigs()
if __name__ == '__main__':
    config.print_config()