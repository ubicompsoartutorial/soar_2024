from simclr.arguments_dict import load_args
from simclr.pretrainer import learn_model
from simclr.utils import set_all_seeds

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = load_args()
    set_all_seeds(args['random_seed'])
    print(args)

    # Starting the pre-training
    learn_model(args=args)

    print('------ Pre-training complete! ------')
