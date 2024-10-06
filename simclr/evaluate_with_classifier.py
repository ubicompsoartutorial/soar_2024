import os
import torch
import torch.nn as nn
from time import time
from torch import optim
from torch.optim.lr_scheduler import StepLR

from simclr.arguments_dict import load_args
from simclr.dataset import load_har_dataset
from simclr.meter import RunningMeter, BestMeter
from simclr.model import Classifier
from simclr.utils import save_meter, compute_best_metrics, compute_metrics, \
    model_save_name, set_all_seeds
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------


def evaluate_with_classifier(args=None):
    """
    Evaluating the performance of the trained SimCLR models with a
    classifier
    :param args: arguments passed
    :return: None
    """
    # Getting the trained model name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['saved_model_folder'] is not None:
        args['saved_model'] = os.path.join(
            dir_path,
            'saved_weights',
            args['saved_model_folder'],
            model_save_name(args) + '.pkl'
        )
    else:
        args['saved_model'] = None

    print(args)

    # Load the target data
    data_loaders, dataset_sizes = load_har_dataset(args, pretrain=False)

    # Tracking meters
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = Classifier(args).to(args['device'])

    # Loading pre-trained weights if available
    if args['saved_model'] is not None:
        model.load_pretrained_weights(args)

    # Optimizer settings
    optimizer = optim.AdamW(model.parameters(),
                            lr=args['classifier_lr'],
                            weight_decay=args['classifier_wd'])
    scheduler = StepLR(optimizer,
                       step_size=10,
                       gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(0, args['num_epochs'])):
        since = time()
    
        # Training
        model, optimizer, scheduler = train(model,
                                            data_loaders["train"],
                                            criterion,
                                            optimizer,
                                            scheduler,
                                            args,
                                            epoch,
                                            dataset_sizes["train"],
                                            running_meter)

        # Validation
        evaluate(model,
                 data_loaders["val"],
                 args,
                 criterion,
                 epoch,
                 phase="val",
                 dataset_size=dataset_sizes["val"],
                 running_meter=running_meter)

        # Evaluating on the test data
        evaluate(model,
                 data_loaders["test"],
                 args,
                 criterion,
                 epoch,
                 phase="test",
                 dataset_size=dataset_sizes["test"],
                 running_meter=running_meter)

        # Saving the logs
        save_meter(args, running_meter, classifier=True)

    # Computing the best metrics
    best_meter = compute_best_metrics(running_meter, best_meter,
                                      classifier=True)
    running_meter.update_best_meter(best_meter)
    save_meter(args, running_meter, classifier=True)

    # Printing the best metrics corresponding to the highest validation
    # F1-score
    best_meter.display()

    return


def train(model, data_loader, criterion, optimizer, scheduler, args, epoch,
          dataset_size, running_meter):
    # Setting the model to training mode
    model.train()

    # Set only softmax layer to trainable
    if args['learning_schedule'] == 'last_layer':
        model.freeze_two_conv_layers()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args['device'])
        labels = labels.long().to(args['device'])

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    scheduler.step()

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_metrics(actual_labels,
                        pred_labels,
                        'train',
                        running_meter,
                        loss,
                        epoch)

    return model, optimizer, scheduler


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter):
    # Setting the model to eval mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args['device'])
        labels = labels.long().to(args['device'])

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_metrics(actual_labels,
                        pred_labels,
                        phase,
                        running_meter,
                        loss,
                        epoch)

    return


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = load_args()
    set_all_seeds(args['random_seed'])
    print(args)

    evaluate_with_classifier(args=args)

    print('------ Evaluation complete! ------')
