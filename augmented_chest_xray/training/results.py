import numpy as np
import torch


class Results:
    """Store results during model training and save them to file."""

    def __init__(self, num_epochs: int, save_path: str):
        self.num_epochs = num_epochs
        self.save_path = save_path + '_results.pt'
        self.epoch = -1
        self.best_epoch = -1
        self.results_dict = {
            'training loss': np.ones(num_epochs)*(-1),
            'validation loss': np.ones(num_epochs)*(-1),
            'mean auroc': np.ones(num_epochs)*(-1),
            'mean F1 score': np.ones(num_epochs)*(-1)}
        self.best_val_loss = np.inf

    def update(
            self, train_loss: float, val_loss: float, auroc: np.ndarray,
            f1: np.ndarray):
        self.epoch += 1
        mean_auroc = np.mean(auroc)
        mean_f1 = np.mean(f1)
        self.results_dict['training loss'][self.epoch] = train_loss
        self.results_dict['validation loss'][self.epoch] = val_loss
        self.results_dict['mean auroc'][self.epoch] = mean_auroc
        self.results_dict['mean F1 score'][self.epoch] = mean_f1
        if val_loss < self.best_val_loss:
            self._update_best_results(val_loss, mean_auroc, mean_f1)
        torch.save(self.results_dict, self.save_path)

    def _update_best_results(
            self, val_loss: float, mean_auroc: float, mean_f1: float):
        self.results_dict['best validation loss'] = val_loss
        self.results_dict['best model mean AUROC'] = mean_auroc
        self.results_dict['best model mean F1 score'] = mean_f1
        self.best_val_loss = val_loss
        self.best_epoch = self.epoch

    def print_summary(self):
        print(
            f'Best model results:\n'
            f'Validation loss: '
            f' {self.results_dict["best validation loss"]:.4f}, '
            f'mean AUROC: {self.results_dict["best model mean AUROC"]:.4f}, '
            f'mean F1 score: '
            f'{self.results_dict["best model mean F1 score"]:.4f}')

    def get_results_dict(self):
        return self.results_dict
