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

        # Pre-training parameters
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 50,
        'patience': 15,

        # Classification parameters
        'classifier_lr': 1e-3,
        'classifier_wd': 0.0,
        'classifier_batch_size': 256,
        'saved_model_folder': '/',
        'learning_schedule': 'last_layer',  # all_layers
        'classification_model': 'mlp',  # linear

        # Reproducibility
        'random_seed': 0,

        # CUDA
        'device': torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu"),
    }

    # Updating the args if we want to change the location of the prepared
    # data and saved model etc.
    if updated_args is not None:
        args['root_dir'] = updated_args['root_dir']

    return args
