from typing import Callable, Tuple, Dict, TYPE_CHECKING
import glob

import webdataset as wds
import torch.utils.data
import torchvision.models
import torch.nn as nn

import transforms.augmented_transforms as transforms
import config

if TYPE_CHECKING is True:
    import torch
    import numpy


dataset_dir = config.DATASET_DIR
labels_dir = config.LABELS_DIR
labels_filenames = {
    'train': config.TRAIN_LABELS,
    'val': config.VAL_LABELS,
    'test': config.TEST_LABELS
}
supported_phases = ['train', 'val', 'test']
supported_models = ['densenet121', 'densenet161', 'resnet50']
n_classes = config.N_CLASSES
debug_sizes = config.DEBUG_SIZES


def create_dataloader(
        phase: str, preproc: Callable[['numpy.ndarray'], 'torch.Tensor'],
        batch_size: int, num_workers: int = 1, debug: bool = False
        ) -> Tuple[Dict, Dict]:
    """Create dataloaders for using in the different phases."""
    if not isinstance(phase, str):
        raise TypeError(f'Expected str, got {type(phase)}.')
    if not phase in supported_phases:
        raise ValueError(f'Expected values are {*supported_phases,}')

    # Configure the datasets and pass them to the dataloaders.
    dataset_filenames, dataset_size = get_dataset_info(phase)
    if debug is True:
        dataset_size = debug_sizes[phase]
    dataset = (
        wds.WebDataset(
            dataset_dir + dataset_filenames, shardshuffle=True)
        .shuffle(1000)
        .decode('rgb')
        .to_tuple("png", "pyd")
        .map_tuple(preproc, transforms.identity)
    )
    if debug is True:
        dataset.with_length(dataset_size)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers
    )

    return dataloader, dataset_size


def load_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Load a pretrained model and replace the classifier layer.
    
    Load a pretrained model and replace the original fully connected
    layer with a fully connected layer that produces a 14-dimensional
    output followed by an elementwise sigmoid function.
    """
    if not isinstance(model_name, str):
        raise TypeError(f'Expected str, got {type(model_name)}.')
    if not model_name in supported_models:
        raise ValueError(f'Expected values are {*supported_models,}')

    if model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(1024, n_classes), nn.Sigmoid()
        )
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(2208, n_classes), nn.Sigmoid()
        )
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Sequential(nn.Linear(2048, n_classes), nn.Sigmoid())

    return model


def get_dataset_info(phase: str):
    """Get the size of the dataset and the files containing it."""
    n_files = len(glob.glob(dataset_dir + f'*{phase}_*')) - 1
    dataset_filenames = (f'ChestXray14_{phase}_{{000..{n_files:03}}}.tar')
    with open(labels_dir + labels_filenames[phase]) as f:
        dataset_size = sum(1 for line in f) - 1
    dataset_names = dataset_filenames
    dataset_size = dataset_size
    return dataset_names, dataset_size
