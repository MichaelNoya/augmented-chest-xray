"""Transforms used for training, validation and dataset creation.

Data augmentation parameters are used for the augmentation applied to
the training set during the training phase. They can be adjusted in the
block titled 'Data augmentation parameters'.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


"""Data augmentation parameters.

Used for augmentation applied to the training set during training.

pre_crop_size: Resize the image to a square of this size before
    center-cropping to 224x224px.
p_train: Probability that the given transform is applied. Used for
    ShiftScaleRotate(), Perspective() and RandomBrightnessContrast().
p_hflip: Probability that HorizontalFlip() is applied.

The remaining parameters are used as arguments in ShiftScaleRotate(),
Perspective() and RandomBrightnessContrast(). Documentation and default
values for these parameters can be found on:
    albumentations.ai/docs/getting_started/transforms_and_targets/
"""
pre_crop_size = 256
p_train = 0.5
p_hflip = 0.5
shift_limit = 0.0625
scale_limit = ((-0.2, 0.1))
rotate_limit = 10
scale = (0.1)
brightness_limit = (-0.2, 0.2)
contrast_limit = (-0.2, 0.2)
pad_mode = cv2.BORDER_CONSTANT
pad_val = (105/256, 105/256, 105/256)

# Constants for required normalization and minimum input size supported
# by DenseNet121.
# DO NOT CHANGE THESE CONSTANTS.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
FINAL_SIZE = 224

# Transform applied during dataset creation.
# The value can be adjusted in the config.py module.
dataset_image_size = config.DATASET_IMAGE_SIZE


def transforms_train(img):
    """Transform images of the training set."""
    transforms = A.Compose([
        A.Resize(height=FINAL_SIZE, width=FINAL_SIZE),
        A.HorizontalFlip(p=p_hflip),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=True)
    ])
    return transforms(image=img)['image']


def transforms_train_augmented(img):
    """Transform and apply augmentation to the training set images."""
    transforms = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=scale_limit,
            rotate_limit=rotate_limit, border_mode=pad_mode, value=pad_val,
            p=p_train
        ),
        A.Perspective(
            scale=scale, pad_mode=pad_mode, pad_val=pad_val, p=p_train
        ),
        A.Resize(height=pre_crop_size, width=pre_crop_size),
        A.RandomCrop(height=FINAL_SIZE, width=FINAL_SIZE),
        A.HorizontalFlip(p=p_hflip),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit, contrast_limit=contrast_limit,
            p=p_train
        ),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=True)
    ])
    return transforms(image=img)['image']


def transforms_test(img):
    """Transform images of the validation set and test set."""
    transforms = A.Compose([
        A.Resize(height=FINAL_SIZE, width=FINAL_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=True)
    ])
    return transforms(image=img)['image']


def transforms_test_centercrop(img):
    """Transform and center crop images of validation and test set."""
    transforms = A.Compose([
        A.Resize(height=pre_crop_size, width=pre_crop_size),
        A.CenterCrop(height=FINAL_SIZE, width=FINAL_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=True)
    ])
    return transforms(image=img)['image']


def transforms_dataset_creation(img):
    """Transform images during dataset creation."""
    transforms = A.Resize(height=dataset_image_size, width=dataset_image_size)
    return transforms(image=img)['image']


def identity(x):
    return x
