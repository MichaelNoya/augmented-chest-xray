import pandas as pd
import numpy as np
import cv2
import webdataset as wds
from PIL import Image # Required by webdataset for png encoding

import transforms.augmented_transforms as transforms
import config


image_dir = config.IMAGE_DIR
labels_dir = config.LABELS_DIR
labels_path = {
    'train': labels_dir + config.TRAIN_LABELS,
    'val': labels_dir + config.VAL_LABELS,
    'test': labels_dir + config.TEST_LABELS
}
dataset_dir = config.DATASET_DIR


def create_dataset(subset: str):
    """Create WebDataset tar archives for sequential data access.
    
    Resize the images and store them in a collection of WebDataset tar
    archives for sequential data access. The archives are saved in the 
    directory defined in config.py under DATASET_DIR.

    Args:
        subset: The subset to be created. The filenames of the images in
        each subset along with their labels are stored in csv files in
        the directory defined in config.py under LABELS_DIR.
    """
    if type(subset) is not str:
        raise TypeError('Subset must be of type string.')
    if subset not in ['train', 'val', 'test']:
        raise ValueError('No such subset.')
    label_df = pd.read_csv(labels_path[subset])
    len_df = label_df.shape[0]

    dataset_filename = dataset_dir + 'ChestXray14_' + subset + '_%03d.tar'
    sink = wds.ShardWriter(dataset_filename, maxcount=1000)
    sink.verbose = 0

    for row in label_df.itertuples():
        print(f'\rAdding image {row[0]}/{len_df-1}', end='')
        img_name = row[1][:-4]
        image = cv2.imread(image_dir + row[1])
        image = transforms.transforms_dataset_creation(img=image)
        labels = np.array(row[2:]).astype('float')
        sink.write({
            "__key__": img_name,
            "png": image,
            "pyd": labels
        })
        
    sink.close()
