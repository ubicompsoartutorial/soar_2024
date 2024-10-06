import copy
import itertools
import torch
from lightly.loss import NTXentLoss
from time import time
from tqdm.auto import tqdm

import simclr.transformations as transformations
from simclr.dataset import load_har_dataset
from simclr.meter import RunningMeter, BestMeter
from simclr.model import SimCLR
from simclr.utils import compute_best_metrics, update_loss, save_meter, \
    save_model, set_all_seeds


def learn_model(args=None):
    print('Starting the pre-training')
    print(args)

    # Setting seed once again
    set_all_seeds(args['random_seed'])

    # Data loaders
    data_loaders, dataset_sizes = load_har_dataset(args, pretrain=True)

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = SimCLR(args).to(args['device'])
    print(model)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args['learning_rate'],
                                weight_decay=args['weight_decay'],
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args['num_epochs']
    )
    criterion = NTXentLoss(temperature=0.1)
    print(criterion)

    trigger_times = 0

    # List of transformations
    # Choosing the best combination as per:
    # https://arxiv.org/pdf/2011.11542.pdf, which is channel shuffled and
    # permuted (7 vs 6)
    transform_funcs_vectorized = [
        transformations.noise_transform_vectorized,
        transformations.scaling_transform_vectorized,
        transformations.rotation_transform_vectorized,
        transformations.negate_transform_vectorized,
        transformations.time_flip_transform_vectorized,
        # transformations.time_segment_permutation_transform_improved,
        transformations.time_warp_transform_low_cost,
        transformations.channel_shuffle_transform_vectorized
    ]

    for epoch in tqdm(range(0, args['num_epochs'])):
        since = time()

        # Training
        model, optimizer = train(model,
                                 data_loaders["train"],
                                 criterion,
                                 optimizer,
                                 args,
                                 epoch,
                                 dataset_sizes["train"],
                                 running_meter,
                                 transform_funcs_vectorized
                                 )

        scheduler.step()

        # Evaluating on the validation data
        evaluate(model,
                 data_loaders["val"],
                 args,
                 criterion,
                 epoch,
                 phase="val",
                 dataset_size=dataset_sizes["val"],
                 running_meter=running_meter,
                 transform_funcs_vectorized=transform_funcs_vectorized
                 )

        # Saving the logs
        save_meter(args, running_meter)

        # Doing the early stopping check
        if epoch >= 2:
            if running_meter.loss['val'][-1] > best_meter.loss["val"]:
                trigger_times += 1

                if trigger_times >= args['patience']:
                    # print('Early stopping the model at epoch: {}. The '
                    #       'validation loss has not improved for {}'.format(
                    #     epoch, trigger_times))
                    break
            else:
                trigger_times = 0
                # print('Resetting the trigger counter for early stopping')

        # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            print('Updating the best val loss at epoch: {}, since {} < '
                  '{}'.format(epoch, running_meter.loss["val"][-1],
                              best_meter.loss["val"]))
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)

            best_model_wts = copy.deepcopy(model.state_dict())

            # Saving the logs
            save_meter(args, running_meter)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Saving the best performing model
    save_model(model, args, epoch=epoch)

    return


def train(model, data_loader, criterion, optimizer, args, epoch, dataset_size,
          running_meter, transform_funcs_vectorized):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0

    # All combinations of transformations
    trans_comb = []

    # Iterating over the data
    for inputs, _ in data_loader:
        if len(trans_comb) == 0:
            trans_comb = [i for i in itertools.permutations
                          (range(len(transform_funcs_vectorized)), 2)]

        # Getting each transform pair
        i1, i2 = trans_comb.pop()
        t1 = transform_funcs_vectorized[i1]
        t2 = transform_funcs_vectorized[i2]

        # changing back to numpy here
        inputs = inputs.cpu().data.numpy()

        # Transforming the input batch two-ways
        data_1 = torch.from_numpy(t1(inputs).copy()).float().to(args['device'])
        data_2 = torch.from_numpy(t2(inputs).copy()).float().to(args['device'])

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs_1 = model(data_1)
            outputs_2 = model(data_2)

            loss = criterion(outputs_1, outputs_2)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.shape[0]

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase="train",
                running_meter=running_meter,
                loss=loss,
                epoch=epoch)

    return model, optimizer


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter, transform_funcs_vectorized):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0

    # All combinations of transformations
    trans_comb = []

    # Iterating over the data
    for inputs, _ in data_loader:
        if len(trans_comb) == 0:
            trans_comb = [i for i in itertools.permutations
                          (range(len(transform_funcs_vectorized)), 2)]

        # Getting each transform pair
        i1, i2 = trans_comb.pop()
        t1 = transform_funcs_vectorized[i1]
        t2 = transform_funcs_vectorized[i2]

        inputs = inputs.cpu().data.numpy()

        # Transforming the input batch two-ways
        data_1 = torch.from_numpy(t1(inputs).copy()).float().to(args['device'])
        data_2 = torch.from_numpy(t2(inputs).copy()).float().to(args['device'])

        with torch.set_grad_enabled(False):
            outputs_1 = model(data_1)
            outputs_2 = model(data_2)

            loss = criterion(outputs_1, outputs_2)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.shape[0]

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase=phase,
                running_meter=running_meter,
                loss=loss,
                epoch=epoch)

    return
