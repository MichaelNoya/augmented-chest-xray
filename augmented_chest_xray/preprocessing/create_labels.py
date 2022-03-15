from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config


def create_labels(
        train_val_test_ratio: Tuple[float, float, float],
        random_seed: int = config.RANDOM_SEED, subset_size: int = None):
    """Create one-hot encoded labels for the train, val and test set."""
    meta_dir = Path(config.META_DATA_DIR)
    meta_path = meta_dir / config.META_FILENAME

    # Define paths for output.
    label_dir = Path(config.LABELS_DIR)
    train_labels_path = label_dir / config.TRAIN_LABELS
    val_labels_path = label_dir / config.VAL_LABELS
    test_labels_path = label_dir / config.TEST_LABELS
        
    # Define which of the input columns to keep and define names of the
    # output columns.
    use_columns = config.USE_COLUMNS
    filename, findings, patient_id = use_columns
    sorted_labels = config.LABELS

    # Load data from the csv file and one-hot encode the labels.
    df = pd.read_csv(meta_path, usecols=use_columns)
    df = pd.concat(
        [df[filename], (df[findings].str.get_dummies()), df[patient_id]],
        axis=1
    )
    df = df[[filename, *sorted_labels, patient_id]]

    # If a subset_size is specified, create labels of only a subset of
    # the patients from the original dataset.
    if subset_size is not None:
        _, df = split_train_test(
            df=df, test_size=subset_size, groups=df[patient_id],
            random_seed=random_seed
        )

    # Split the labels into train, val and test set and save to file.
    test_size = train_val_test_ratio[2]
    trainval_df, test_df = split_train_test(
        df=df, test_size=test_size, groups=df[patient_id],
        random_seed=random_seed
    )
    val_size = train_val_test_ratio[1]/(1 - train_val_test_ratio[2])
    train_df, val_df = split_train_test(
        df=trainval_df, test_size=val_size,
        groups=trainval_df[patient_id], random_seed=random_seed
    )
    train_df.drop(columns=patient_id).to_csv(train_labels_path, index=False)
    val_df.drop(columns=patient_id).to_csv(val_labels_path, index=False)
    test_df.drop(columns=patient_id).to_csv(test_labels_path, index=False)


def split_train_test(
        df: pd.DataFrame, test_size: float, groups: pd.Series,
        random_seed: int = None):
    """Split the data into train and test sets without group overlap."""
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[trainval_idx, :].reset_index(drop=True)
    test_df = df.iloc[test_idx, :].reset_index(drop=True)
    return train_df, test_df
