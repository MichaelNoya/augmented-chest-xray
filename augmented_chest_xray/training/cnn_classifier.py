from typing import Dict, Tuple, TYPE_CHECKING
from pathlib import Path
import time

import numpy as np
import torch

import training.metrics as metrics
import training.results as results

if TYPE_CHECKING is True:
    import torch.nn as nn
    import torch.utils.data
    import torch.optim


class ImageClassifier:
    """Wrap a model for training and validation.
    
    Wrap a pretrained torchvision model for finetuning and validation
    on data from the NIH Chest X-ray Dataset dataset of 14 Common Thorax
    Disease Categories. 
    
    Methods:
        train(): Train a model or continue a previous training session.
        validate(): Validate a trained model on a test set.
    """

    def __init__(self, model: 'nn.Module', dict_load_path: str = None):
        """Load a model and optional state_dict."""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        if dict_load_path is not None:
            self._load_model(dict_load_path=dict_load_path)

    def train(
            self, dataloaders: Dict[str, 'torch.utils.data.DataLoader'], 
            dataset_sizes: Dict[str, int], criterion: 'nn.Module',
            optimizer: 'torch.optim.Optimizer', scheduler, num_epochs: int,
            save_dir: str, debug: bool = False):
        """Train the model on a dataset.
        
        Args:
            dataloaders: Dict of DataLoaders with keys 'train' and
                'val', containing the training and validation data.
            dataset_sizes: Dict with the sizes of the datasets contained
                in the dataloaders.
            criterion: Loss function used for training and validation.
            optimizer: Optimization algorithm used for training.
            scheduler: Scheduler that can adjust the learning rate of
                the optimizer.
            num_epochs: Number of epochs to train.
            save_dir: Directory for saving the state_dict of the best
                model along with the corresponding thresholds. Also used
                for saving a results_dict containing training and
                validation loss and metrics.
            debug: If True the sizes of the datasets are reduced to the
                values specified in config.py under DEBUG_SIZES.            
        """
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes 
        self.batches_per_epoch = {
            phase: np.ceil(dataset_sizes[phase]/dataloaders[phase].batch_size)
            for phase in ['train', 'val']
        }
        self.criterion = criterion
        self.debug = debug

        # Define the paths for saving the trained model with
        # corresponding thresholds and the results dictionary.
        if not Path(save_dir).is_dir():
            raise ValueError('Save directory not found.')
        save_path = save_dir + f'{time.strftime("%Y_%m%d_%H%M%S")}'
        if debug is True:
            save_path = save_path + '_debug'
        print(
            f'Best model and results will be saved to:\n'
            f'{save_path}_model.pt\n{save_path}_results.pt\n')

        # Create a Results object to log the loss and metrics.
        TrainingResults = results.Results(
            num_epochs=num_epochs, save_path=save_path)

        # Start the training.    
        epoch = 0
        since = time.time()
        while epoch < num_epochs:
            print(f'Epoch {epoch + 1}/{num_epochs}')

            # Training phase
            phase = 'train'
            self.model.train(mode=True)
            running_loss = 0.0

            # Epoch labels and epoch predictions are used to calculate
            # the thresholds.
            epoch_labels = torch.FloatTensor().to(self.device)
            epoch_predictions = torch.FloatTensor().to(self.device)

            for idx, [images, labels] in enumerate(self.dataloaders[phase]):
                print(
                    f'\rTraining batch {idx + 1}/'
                    f'{int(self.batches_per_epoch[phase])}', end='')

                images = images.to(self.device)
                labels = labels.to(self.device).float()
                optimizer.zero_grad()
                predictions, loss = self._predict(images, labels, phase)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()*images.size(0)
                with torch.no_grad():
                    epoch_labels = torch.cat((epoch_labels, labels), 0)
                    epoch_predictions = torch.cat(
                        (epoch_predictions, predictions), 0)

                if debug is True and idx + 2 > self.batches_per_epoch[phase]:
                    break

            # Calculate the training loss and the thresholds.
            train_loss = running_loss/dataset_sizes[phase]
            epoch_labels = epoch_labels.cpu().numpy()
            epoch_predictions = epoch_predictions.cpu().numpy()
            self.thresholds = metrics.get_thresholds(
                labels=epoch_labels, predictions=epoch_predictions)
            print(f'\rTraining loss: {train_loss:.4f} ')

            # Validation phase.
            val_loss, auroc, f1 = self._validate()

            # Save the training and validation loss and the metrics.
            TrainingResults.update(
                train_loss=train_loss, val_loss=val_loss, auroc=auroc, f1=f1)
            print(
                f'\rValidation loss: {val_loss:.4f}, '
                f'mean AUROC: {np.mean(auroc):.4f}, '
                f'mean F1 score: {np.mean(f1):.4f}')
            scheduler.step(val_loss)

            # If the current epoch produces a lower validation loss than
            # previous epochs, the model and corresponding thresholds
            # are saved to file.
            if epoch == TrainingResults.best_epoch:
                print('Best val_loss. Saving model to file... ', end='')
                self._save_model(save_path=save_path)
                print('saved successfully.')

            time_elapsed = time.time() - since
            print(
                f'Time elapsed: {time_elapsed//60:.0f}m '
                f'{time_elapsed % 60:.0f}s\n')
            epoch += 1

        # Training complete
        print(f'Training complete.')
        TrainingResults.print_summary()

    def validate(
            self, dataloader = None, dataset_size: int = None,
            debug: bool = False
            ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Validate the model on a given dataset.

        Args:
            dataloader: DataLoader containing the validation data.
            dataset_size: Number of samples in the validation set.
            debug: If True the sizes of the datasets are reduced to the
                values specified in config.py under DEBUG_SIZES. 

        Returns:
            AUROC, mean AUROC, F1 score and mean F1 score.
        """
        try:
            self.dataloaders
        except AttributeError:
            if dataloader is None:
                raise TypeError('No dataloader provided.')                
            elif dataset_size is None:
                raise TypeError('No dataset_size provided.')
            else:
                self.dataloaders = {'val': dataloader}
                self.dataset_sizes = {'val': dataset_size}
                self.batches_per_epoch = {
                    'val': np.ceil(dataset_size/dataloader.batch_size)}
        self.debug = debug

        _, auroc, f1 = self._validate()

        return auroc, np.mean(auroc), f1, np.mean(f1)

    def _predict(self, images, labels, phase):
        with torch.set_grad_enabled(phase == 'train'):
            predictions = self.model(images)
            try:
                loss = self.criterion(predictions, labels)
            except AttributeError:
                loss = torch.tensor(0)      
        return predictions, loss

    def _validate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        phase = 'val'
        self.model.train(mode=False)
        running_loss = 0.0

        # Epoch labels and predictions are used to calculate validation
        # metrics
        epoch_labels = torch.FloatTensor().to(self.device)
        epoch_predictions = torch.FloatTensor().to(self.device)

        for idx, [images, labels] in enumerate(self.dataloaders[phase]):
            print(
                f'\rValidation batch {idx + 1}/'
                f'{int(self.batches_per_epoch[phase])}', end='')

            images = images.to(self.device)
            labels = labels.to(self.device).float()
            predictions, loss = self._predict(images, labels, phase)

            running_loss += loss.item()*images.size(0)
            with torch.no_grad():
                epoch_labels = torch.cat((epoch_labels, labels), 0)
                epoch_predictions = torch.cat(
                    (epoch_predictions, predictions), 0)

            if self.debug is True and idx + 2 > self.batches_per_epoch[phase]:
                break

        # Calculate loss and metrics
        val_loss = running_loss/self.dataset_sizes[phase]
        epoch_labels = epoch_labels.cpu().numpy()
        epoch_predictions = epoch_predictions.cpu().numpy()
        auroc = metrics.calculate_auroc(epoch_labels, epoch_predictions)
        thresholded_predictions = 1*(epoch_predictions > self.thresholds)
        f1 = metrics.calculate_f1(
                labels=epoch_labels, predictions=thresholded_predictions)

        return val_loss, auroc, f1

    def _save_model(self, save_path: str):
        """Save a dict containing the state_dict and thresholds."""
        model_dict = {
            'model': self.model.state_dict(),
            'thresholds': self.thresholds}
        torch.save(model_dict, save_path + '_model.pt')

    def _load_model(self, dict_load_path: str) -> Dict:
        """Load a dict containing a state_dict and thresholds."""
        if not Path(dict_load_path).is_file():
            raise ValueError('Model state_dict not found.')
        model_dict = torch.load(dict_load_path, map_location=self.device)
        self.model.load_state_dict(model_dict['model'])
        self.thresholds = model_dict['thresholds']
        print('Model state_dict and thresholds loaded successfully.')
