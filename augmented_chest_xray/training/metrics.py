from sklearn.metrics import f1_score, roc_curve, roc_auc_score
import numpy as np

import config


classes = config.LABELS
n_classes = len(classes)


def check_input(labels: np.ndarray, predictions: np.ndarray):
    """Validate the input arrays."""
    if labels.ndim != 2 or predictions.ndim != 2:
        raise TypeError(
            f'Expected two two-dimensional arrays, got {labels.ndim} '
            f'and {predictions.ndim} dimensions.'
        )
    if labels.shape[1] != predictions.shape[1] or labels.shape[1] != n_classes:
        raise TypeError(
            f'Expected two arrays with {n_classes} classes each, got '
            f'{labels.shape[1]} and {predictions.shape[1]}.'
        )
    if labels.shape[0] != predictions.shape[0]:
        raise TypeError(
            'Expected two arrays with the same number of observations, got '
            f'{labels.shape[0]} and {predictions.shape[0]}.'
        )


def calculate_auroc(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculate the AUROC for each class for two arrays."""
    check_input(labels, predictions)
    auroc = np.zeros(n_classes)
    for patho in range(n_classes):
        try:
            auroc[patho] = roc_auc_score(
                labels[:, patho], predictions[:, patho])
        except ValueError:
            # The AUC for a class is not defined if the corresponding
            # labels all have the same value. This can happen when 
            # working with a subset of the dataset, e.g., for debugging.
            # If this subset is chosen too small it might not contain
            # positive examples of a given class.
            auroc[patho] = np.nan
    return auroc


def calculate_f1(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculate the F1 score for each class for two arrays."""
    check_input(labels, predictions)
    f1 = np.zeros(n_classes)
    for patho in range(n_classes):
        f1[patho] = f1_score(
            labels[:, patho], predictions[:, patho], labels=[1, 0],
            zero_division=0)
    return f1


def get_thresholds(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculate the optimal thresholds for each class."""
    check_input(labels, predictions)
    opt_thresholds = np.zeros(n_classes)
    for patho in range(n_classes):
        fpr, tpr, thresholds = roc_curve(
            labels[:, patho], predictions[:, patho], pos_label=1)
        idx = np.argmax(tpr-fpr)
        opt_thresholds[patho] = thresholds[idx]
        if thresholds[idx] >= 1:
            print(
                f'Warning: Optimal calculated threshold for {classes[patho]} '
                f'is {thresholds[idx]:.4f}. Using 0.5 as threshold instead.'
            )
            opt_thresholds[patho] = 0.5
    return opt_thresholds
