class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'Indian_pines' # Indian_pines/indian_pines_corrected/PaviaU/PaviaCenter
    train_percent = 0.2

    # train/test parameters
    model_name = 'SimpleFC' # Pyramidal_ResNet/SimpleFC/PPFsNet
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 120
    step_size = 40
    batch_size = 100
    seed = 75
    lr = 0.1
    weight_decay = 1e-4 # 1e-4

    # IndianPines data preparation parameters
    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
    patch_size = 21
    if dataset == 'Indian_pines':
        indianPines_band = 220
    elif dataset == 'Indian_pines_corrected':
        indianPines_band = 200
    indianPines_class = 16
    indianPines_seed = 6

    # Dataset
    if dataset == 'Indian_pines' or dataset == 'Indian_pines_corrected':
        band = indianPines_band
        num_classes = indianPines_class
    
    # SimpleNet parameters
    conv1 = 500
    conv2 = 100
    fc1 = 200
    fc2 = 84

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
          '\tpatch_size: {}\n'
          '\tdataset: {}\n'
          '\tband: {}\n'
          '\tnumber_classes: {}\n'
          
          '# SimpleNet parameters\n'
          '\tconv1: {}\n'
          '\tconv2: {}\n'
          '\tfc1: {}\n'
          '\tfc2: {}\n'
          
          '# SimpleFC parameters\n'
          '\tFC_1: {}\n'
          '\tFC_2: {}\n'
          '\tFC_3: {}\n'
          '\tFC_4: {}\n'.format(
          config.cuda, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.patch_size, config.dataset,
          config.band, config.num_classes,
          config.conv1, config.conv2,config.fc1, config.fc2,
          config.FC_1, config.FC_2, config.FC_3, config.FC_4
          ))
        print('#'*60)
        print('\n')

config = DefaultConfigs()
if __name__ == '__main__':
    config.print_config()