import os

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_args(updated_args=None):
    args = {
        # Data loading parameters
        'window': 100,
        'overlap': 50,
        'input_size': 3,

        # Dataset parameters
        'dataset': 'motionsense',
        'root_dir': '',
        'data_file': 'motionsense.pkl',
        'num_classes': 6,

        # Classification parameters
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'num_epochs': 50,

        'classifier_lr': 1e-4,
        'classifier_wd': 1e-4,

        'classification_model': 'mlp',

        # Reproducibility
        'random_seed': 42,

        # CUDA
        'device': torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu"),
    }

    # Updating the args if we want to change the location of the prepared
    # data and saved model etc.
    if updated_args is not None:
        args['root_dir'] = updated_args['root_dir']

    return args
