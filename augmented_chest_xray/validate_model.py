"""Perform model validation on a test set."""

from typing import Tuple
import argparse
import os
import glob

import torch
import numpy as np
import pandas as pd

import transforms.augmented_transforms as transforms
import training.utils as utils
import training.cnn_classifier as cnn_classifier
import config


def validate_model(
        saved_model_filename: str = None,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        center_crop: bool = config.CENTER_CROP,
        debug: bool = config.DEBUG
        ) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Perform model validation on a test set.
    
    Load the state_dict of a trained CNN image classifier and validate
    the performance on a test set.

    Args:
        saved_model_filename: Name of the file containing the state_dict
            of the model.
        batch_size: Batch size to use during validation.
        num_workers: Number of workers to use.
        center_crop: Center crop the images of the test set for
            validation. Set to True if the dataset was enhanced with
            data augmentation during training.
        debug: Reduce size of dataset for debugging.

    Returns:
        AUROC for each class
        mean AUROC
        F1 score for each class
        mean F1 score
    """

    save_dir = config.SAVE_DIR
    try:
        dict_load_path = save_dir + saved_model_filename
    except TypeError:
        # Load newest file, if no filename is provided
        files = glob.glob(save_dir + '*_model.pt')
        dict_load_path = max(files, key=os.path.getctime)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choose which preprocessing steps to apply to the test set.
    if center_crop is True:
        preproc = transforms.transforms_test_centercrop
    else:
        preproc = transforms.transforms_test

    # Create a dataloader for the test set.
    dataloader, dataset_size = utils.create_dataloader(
        phase='test', preproc=preproc, batch_size=batch_size,
        num_workers=num_workers, debug=debug
    )

    # Load the model.
    print(f'Loading from {os.path.basename(dict_load_path)}.')
    model = utils.load_model(model_name='densenet121')
    model.to(device)
    xray_classifier = cnn_classifier.ImageClassifier(
        model=model, dict_load_path=dict_load_path
    )

    # Perform validation on the given dataset.
    auroc, mean_auroc, f1, mean_f1 = xray_classifier.validate(
        dataloader=dataloader, dataset_size=dataset_size, debug=debug
    )

    return auroc, mean_auroc, f1, mean_f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--saved_model',
        help='Name of file from which to load the state_dict.'
    )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=config.BATCH_SIZE,
        help='Batch size to use.'
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=config.NUM_WORKERS,
        help='Number of workers to use.'
    )
    parser.add_argument(
        '-c', '--center_crop', action='store_true',
        help=(
            'Center crop the images of the test set. Use this if the dataset '
            'was enhanced with data augmentation during training.'
        )
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Reduce size of dataset for debugging.'
    )
    parser.add_argument(
        '-f', '--display_f1', action='store_true',
        help='Display F1 Score in results.'
    )
    args = parser.parse_args()

    auroc, mean_auroc, f1, mean_f1 = validate_model(
        saved_model_filename=args.saved_model,
        batch_size=args.batch_size, num_workers=args.workers,
        center_crop=args.center_crop, debug=args.debug
    )
    print('\nValidation complete.\n')
    data = [[*auroc, mean_auroc], [*f1, mean_f1]]
    columns = [*config.LABELS, 'Mean']
    index = ['AUROC', 'F1 Score']
    df = pd.DataFrame(data=data, index=index, columns=columns)
    if args.display_f1 is True:
        print(df.transpose())
    else:
        print('Values for AUROC:')
        print(df.loc['AUROC'].to_string())
