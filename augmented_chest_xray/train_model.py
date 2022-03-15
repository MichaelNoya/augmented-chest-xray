"""Trains a CNN image classifier on the ChestX-Ray14 dataset."""

import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim

import transforms.augmented_transforms as transforms
import training.utils as utils
import training.cnn_classifier as cnn_classifier
import config


def train_model(
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        num_epochs: int = config.NUM_EPOCHS,
        learning_rate: float = config.LEARNING_RATE,
        scheduler_factor: float = config.SCHEDULER_FACTOR,
        scheduler_patience: int = config.SCHEDULER_PATIENCE,
        data_augmentation: bool = config.DATA_AUGMENTATION,
        debug: bool = True,
        saved_model_filename: str = None):
    """Train a CNN model and save the results to file.
    
    Main module, used to train the model on the ChestX-Ray14 dataset.
    The state_dict of the trained model and the corresponding thresholds
    are stored in a dictionary and saved to file.
    The training and validation loss of each epoch along with the AUROC
    and F1 score are also saved.
    If saved_model_filename is specified, a saved state_dict will be
    loaded to continue a previous training session.

    Args:
        batch_size: Batch size to use during validation.
        num_workers: Number of workers to use.
        num_epochs: Number of epochs to train.
        learning_rate: Learning rate for optimizer initialization.
        scheduler_factor: Factor by which the scheduler should reduce
            the learning rate.
        scheduler_patience: Number of epochs with no improvement that
            the scheduler should wait before decreasing the learning
            rate.
        data_augmentation: Activate data augmentation.
        debug: Reduce size of dataset for debugging.
        saved_model_filename: File to load a state_dict from a previous
            training session.
    """
    save_dir = config.SAVE_DIR
    try:
        dict_load_path = save_dir + saved_model_filename
    except TypeError:
        dict_load_path = None
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # Choose which preprocessing steps to apply to the datasets.
    if data_augmentation is True:
        preproc = {
            'train': transforms.transforms_train_augmented,
            'val': transforms.transforms_test_centercrop
        }
    else:
        preproc = {
            'train': transforms.transforms_train,
            'val': transforms.transforms_test
        }

    # Create dataloaders for the training and validation datasets.
    dataloaders = {}
    dataset_sizes = {}
    for phase in ['train', 'val']:
        dataloaders[phase], dataset_sizes[phase] = utils.create_dataloader(
            phase=phase, preproc=preproc[phase], batch_size=batch_size,
            num_workers=num_workers, debug=debug
        )

    # Load the pretrained model.
    model = utils.load_model(model_name='densenet121', pretrained=True)
    model.to(device)
    xray_classifier = cnn_classifier.ImageClassifier(
        model, dict_load_path=dict_load_path
    )

    # Define the criterion, optimizer, and a learning rate scheduler.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience,
        verbose=True
    )

    # List the used arguments and start the training.
    print(
        f'Training on {device_name.upper()}. '
        f'Batch size: {batch_size}, num workers: {num_workers}.\n'
        f'Learning rate: {learning_rate}, '
        f'lr scheduler factor: {scheduler_factor}, '
        f'lr scheduler patience: {scheduler_patience}.\n'
        f'Data augmentation: '
        f'{"on" if data_augmentation is True else "off"}.\n'
    )
    xray_classifier.train(
        dataloaders=dataloaders, dataset_sizes=dataset_sizes,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        num_epochs=num_epochs, save_dir=save_dir, debug=debug
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--batch_size', type=int, default=config.BATCH_SIZE,
        help='Batch size to use during training and validation.'
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=config.NUM_WORKERS,
        help='Number of workers to use during training and validation.'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=config.NUM_EPOCHS,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '-l', '--learning_rate', type=float, default=config.LEARNING_RATE,
        help='Learning rate for optimizer initialization.'
    )
    parser.add_argument(
        '-f', '--s_factor', type=float, default=config.SCHEDULER_FACTOR,
        help='Factor by which the scheduler should reduce the learning rate.'
    )
    parser.add_argument(
        '-p', '--s_patience', type=int, default=config.SCHEDULER_PATIENCE,
        help=(
            'Number of epochs with no improvement that the scheduler should '
            'wait before decreasing the learning rate.'
        )
    )
    parser.add_argument(
        '-a', '--augmentation', action='store_true',
        help='Activate data augmentation.'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Reduce size of dataset for debugging.'
    )
    parser.add_argument(
        '-s', '--saved_model', default=None,
        help='File to load a state_dict from a previous training session.'
    )
    args = parser.parse_args()
    
    train_model(
        batch_size=args.batch_size, num_workers=args.workers,
        num_epochs=args.epochs, learning_rate=args.learning_rate,
        scheduler_factor=args.s_factor, scheduler_patience=args.s_patience,
        data_augmentation=args.augmentation, debug=args.debug,
        saved_model_filename=args.saved_model
    )
