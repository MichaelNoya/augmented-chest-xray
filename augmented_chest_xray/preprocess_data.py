"""Preprocess the dataset for use in model training and validation."""

import argparse
from pathlib import Path

import preprocessing.create_labels as create_labels
import preprocessing.create_dataset as create_dataset
import config


def preprocess_data(
        random_seed: float = config.RANDOM_SEED, subset_size: int = None):
    """Preprocess the dataset for use in model training and validation.

    The ChestX-Ray14 is divided into training, validation and test set,  
    and stored in a collection of tar archives for sequential data
    access.

    Args:
        random_seed: Seed for the random shuffle split of the dataset.
        subset_size: If a subset_size is specified, the training, 
            validation and test sets will be created using images of
            only a subset of the patients from the original dataset.
    """
    # Create directories for the labels and the datasets.
    label_dir = Path(config.LABELS_DIR)
    if not label_dir.is_dir():
        Path.mkdir(label_dir)
    dataset_dir = Path(config.DATASET_DIR)
    if not dataset_dir.is_dir():
        Path.mkdir(dataset_dir)

    # Create three csv files containing the filenames and one-hot
    # encoded labels of the training, validation and test sets.
    train_val_test_ratio = config.TRAIN_VAL_TEST_RATIO
    create_labels.create_labels(
        train_val_test_ratio=train_val_test_ratio, random_seed=random_seed,
        subset_size=subset_size)

    # Resize the original images and store them with their corresponding 
    # labels as a collection of tar archives for sequential access.
    for subset in ['train', 'val', 'test']:
        dataset_filename = 'ChestXray14_' + subset + '_000.tar'
        dataset_path = dataset_dir / dataset_filename
        print(f'\nCreating {subset} dataset:')
        if dataset_path.is_file():
            print(
                f'File {dataset_filename} already exists, '
                f'skipping {subset} dataset.'
            )
        else:
            create_dataset.create_dataset(subset=subset)
            print(f'\nDataset created.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--random_seed', type=int, default=config.RANDOM_SEED,
        help='Seed for the random shuffle split of the dataset.'
    )
    parser.add_argument(
        '-s', '--subset_size', type=int, default=None,
        help=(
            'If a subset_size is specified, the training, validation and test '
            'sets will be created using images of only a subset of the '
            'patients from the original dataset.'
        )
    )
    args = parser.parse_args()

    preprocess_data(
        random_seed=args.random_seed, subset_size=args.subset_size
    )
