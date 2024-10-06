import os
import torch
import torch.nn as nn
from time import time
from torch import optim
from torch.optim.lr_scheduler import StepLR

from .arguments_dict import load_args
from .dataset import load_har_dataset
from .meter import RunningMeter, BestMeter
from .model import Classifier
from .utils import save_meter, compute_best_metrics, compute_metrics, \
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
    print(args)

    # Load the target data
    data_loaders, dataset_sizes = load_har_dataset(args)

    # Tracking meters
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = Classifier(args).to(args['device'])

    # Optimizer settings
    optimizer = optim.AdamW(model.parameters(),
                            lr=args['learning_rate'],
                            weight_decay=args['weight_decay'])
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
