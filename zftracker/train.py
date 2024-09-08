import math
from time import time

import torch.optim.lr_scheduler
import torch
from IPython.display import clear_output

import torch.nn.functional as F
from torch.utils.data import DataLoader

from .util.tqdm import TQDM as tqdm
from .util.str import format_seconds_to_hms_string
from .util.str import format_scientific
from .visual.plots import plot_metric

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup Cosine Annealing Learning Rate Scheduler
    """

    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
        """
        Warmup Cosine Annealing Learning Rate Scheduler
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            warmup_steps (int): Number of warmup steps. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Get learning rate
        Returns:
            list: Current learning rate.
        """
        if self.last_epoch < self.warmup_steps:
            return [
                self.eta_min + base_lr * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_steps)
                        / (self.T_max - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]


class WarmUpCosineRandaugmentScheduler(object):
    """
    Warmup Cosine Annealing RandAugment Scheduler
    """

    def __init__(
        self,
        dataset,
        n_range,
        m_range,
        total_epochs,
        warmup_epochs,
        augmentation_batch_size,
    ):
        """
        Warmup Cosine Annealing RandAugment Scheduler
        Args:
            dataset (torch.utils.data.Dataset): Dataset to be augmented.
            n_range (tuple): Range of n values for RandAugment.
            m_range (tuple): Range of m values for RandAugment.
            total_epochs (int): Total number of epochs.
            warmup_epochs (int): Number of warmup epochs.
            augmentation_batch_size (int): Batch size for RandAugment. If False or 0, sequential initialization is used.
        """
        self.dataset = dataset
        self.n_range = n_range
        self.m_range = m_range
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.augmentation_batch_size = augmentation_batch_size
        self.latest_n = None
        self.latest_m = None
        self.epoch = 0
        self._step()

    def bounded_cosine(self, x, minimum, maximum):
        """
        Cosine function bounded between minimum and maximum.
        Args:
            x (float): Input value.
            minimum (float): Minimum value.
            maximum (float): Maximum value.
        Returns:
            float: Cosine function bounded between minimum and maximum.
        """
        return minimum + 0.5 * (maximum - minimum) * (
            1 + math.cos(x * math.pi + math.pi)
        )
    
    def _step(self):
        self.step()

    def step(self, verbose=True):
        """
        Step the scheduler forward one epoch.
        Args:
            verbose (bool): Whether to print the current epoch's n and m values.
        """
        # Time
        if verbose:
            start_time = time()

        # Calculate the current epoch's n and m values
        if self.epoch < self.warmup_epochs:
            self.latest_n = self.n_range[0]
            self.latest_m = self.m_range[0]
        else:
            self.latest_n = self.bounded_cosine(
                (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                self.n_range[0],
                self.n_range[1],
            )
            self.latest_m = self.bounded_cosine(
                (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                self.m_range[0],
                self.m_range[1],
            )

        # Initialize the dataset with the new n and m values
        if self.augmentation_batch_size:
            self.dataset.parallel_initialize(
                rangaug_param=(self.latest_n, self.latest_m),
                batch_size=self.augmentation_batch_size
            )
        else:
            self.dataset.sequential_initialize(
                rangaug_param=(self.latest_n, self.latest_m)
            )

        # Increase the epoch count
        self.epoch += 1

        # Clamp n and m values to their maximums after num_epochs
        if self.epoch >= self.total_epochs:
            self.epoch = self.total_epochs

        # Print time
        if verbose:
            print(
                f"RandAugment(n = {self.latest_n:.2f}, m = {self.latest_m:.2f});",
                f"Time: {(time() - start_time):.2f} s",
            )

class WarmUpCosRandaugSchedulerWithAttentionRadius(WarmUpCosineRandaugmentScheduler):
    """
    Warmup Cosine Annealing RandAugment Scheduler with Attention Radius
    """

    def __init__(
        self,
        dataset,
        val_dataset,
        n_range,
        m_range,
        total_epochs,
        warmup_epochs,
        augmentation_batch_size,
        attention_radius_range,
    ):
        """
        Warmup Cosine Annealing RandAugment Scheduler with Attention Radius
        Args:
            dataset (torch.utils.data.Dataset): Dataset to be augmented.
            n_range (tuple): Range of n values for RandAugment.
            m_range (tuple): Range of m values for RandAugment.
            total_epochs (int): Total number of epochs.
            warmup_epochs (int): Number of warmup epochs.
            augmentation_batch_size (int): Batch size for RandAugment. If False or 0, sequential initialization is used.
            attention_radius_range (tuple): Range of attention radius values.
        """
        super().__init__(
            dataset,
            n_range,
            m_range,
            total_epochs,
            warmup_epochs,
            augmentation_batch_size,
        )
        self.attention_radius_range = attention_radius_range
        self.latest_attention_radius = None
        self.val_dataset = val_dataset
        self.step()

    def _step(self):
        pass

    def step(self, verbose=True):
        """
        Step the scheduler forward one epoch.
        Args:
            verbose (bool): Whether to print the current epoch's n, m, and attention radius values.
        """
        # Time
        if verbose:
            start_time = time()

        # Calculate the current epoch's n, m, and attention radius values
        if self.epoch < self.warmup_epochs:
            self.latest_n = self.n_range[0]
            self.latest_m = self.m_range[0]
            self.latest_attention_radius = self.attention_radius_range[0]
        else:
            self.latest_n = self.bounded_cosine(
                (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                self.n_range[0],
                self.n_range[1],
            )
            self.latest_m = self.bounded_cosine(
                (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                self.m_range[0],
                self.m_range[1],
            )
            self.latest_attention_radius = self.bounded_cosine(
                (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                self.attention_radius_range[0],
                self.attention_radius_range[1],
            )

        # Initialize the dataset with the new n and m values
        if self.augmentation_batch_size:
            self.dataset.parallel_initialize(
                rangaug_param=(self.latest_n, self.latest_m),
                attention_radius=self.latest_attention_radius,
                batch_size=self.augmentation_batch_size
            )
            self.val_dataset.parallel_initialize(
                rangaug_param=None,
                attention_radius=self.latest_attention_radius,
                batch_size=self.augmentation_batch_size
            )
        else:
            self.dataset.sequential_initialize(
                rangaug_param=(self.latest_n, self.latest_m),
                attention_radius=self.latest_attention_radius
            )
            self.val_dataset.sequential_initialize(
                rangaug_param=None,
                attention_radius=self.latest_attention_radius
            )

        # Increase the epoch count
        self.epoch += 1

        # Clamp n and m values to their maximums after num_epochs
        if self.epoch >= self.total_epochs:
            self.epoch = self.total_epochs

        # Print time
        if verbose:
            print(
                f"RandAugment(n = {self.latest_n:.2f}, m = {self.latest_m:.2f}); Attention Radius = {self.latest_attention_radius:.2f}",
                f"Time: {(time() - start_time):.2f} s",
            )


class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 device: torch.device,
                 num_epochs: int,
                 loss_function: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 randaug_scheduler: WarmUpCosineRandaugmentScheduler,
                 metrics: list,
                 mode: str = 'regression',
                 plots: dict = {},
                 verbose: bool = True):
        """
        Train a PyTorch model for a specified number of epochs, evaluating on a validation set and metrics.
        Args:
            model (torch.nn.Module): Model to be trained.
            dataloader (torch.utils.data.DataLoader): Training set dataloader.
            val_dataloader (torch.utils.data.DataLoader): Validation set dataloader.
            device (torch.device): Device to run the model on.
            num_epochs (int): Number of epochs to train for.
            loss_function (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            randaug_scheduler (WarmUpCosineRandaugmentScheduler): RandAugment scheduler.
            metrics (list): List of metrics to evaluate on.
            mode (str): 'vector_classification' or 'regression' or 'heatmap_classification'.
            plots (dict): Dictionary of plots to show. Each key is a metric to plot, and each value is a dictionary with keys 'log' and 'alpha'.
            verbose (bool): Whether to print the current epoch's loss and learning rate.
        """
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.randaug_scheduler = randaug_scheduler
        self.metrics = metrics
        self.mode = mode
        self.plots = plots
        self.verbose = verbose

        self.model.to(device)

        # Initialize history dictionary
        self.history = {"lr": [], "randaug_n": [], "randaug_m": []}

    def on_batch_start(self, batch):
        self.history["lr"].append(self.lr_scheduler.get_last_lr()[0])

    def on_batch_end(self, batch):
        self.lr_scheduler.step()

    def on_epoch_start(self, epoch):

        self.history["randaug_n"].append(self.randaug_scheduler.latest_n)
        self.history["randaug_m"].append(self.randaug_scheduler.latest_m)

        if epoch > 0:

            time_per_epoch = (time() - self._last_epoch_start_time)
            remaining_time = (self.num_epochs - epoch) * time_per_epoch

            if "time_per_epoch" not in self.history:
                self.history["time_per_epoch"] = []
            self.history["time_per_epoch"].append(time_per_epoch)

        self._last_epoch_start_time = time()

        if self.verbose:

            print("-" * 10)
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            if epoch > 0:
                print(
                    f"Time per epoch: {format_seconds_to_hms_string(time_per_epoch)}")
                print(
                    f"Estimated remaining time: {format_seconds_to_hms_string(remaining_time)}")

            print("-" * 10)

    def on_epoch_end(self, epoch):
        history_keys = list(self.history.keys())
        for key in history_keys:
            if "train_" in key and "_per_epoch" not in key:
                name = key + "_per_epoch"
                if name not in self.history:
                    self.history[name] = []
                epoch_loss = sum(
                    self.history[key][-len(self.dataloader):]) / len(self.dataloader)
                self.history[name].append(epoch_loss)

        if self.plots or self.verbose:
            clear_output(wait=True)

        if self.plots:
            for key in self.plots:
                plot_metric(self.history,
                            key,
                            show_optimum=self.plots[key].get('show_optimum', 'auto'),
                            log=self.plots[key].get('log', 'auto'),
                            alpha=self.plots[key].get('alpha', 1),
                            ylim_lower=self.plots[key].get('ylim_lower', 'auto'),
                            ylim_upper=self.plots[key].get('ylim_upper', 'auto'),
                            format=self.plots[key].get('format', 'auto'))

        if self.verbose:
            print(
                f"Loss: {format_scientific(epoch_loss)}, Learning rate: {format_scientific(self.lr_scheduler.get_last_lr()[0])}"
            )

        if epoch < self.num_epochs - 1:
            self.randaug_scheduler.step()

    def handle_train_loss(self, loss):
        # Loss may be dict or tensor
        if isinstance(loss, dict):
            # Append losses to the history
            for key in loss:
                name = "train_"+key
                if name not in self.history:
                    self.history[name] = []
                self.history[name].append(loss[key].item())
            loss = loss["loss"]
        elif isinstance(loss, torch.Tensor):
            if "train_loss" not in self.history:
                self.history["train_loss"] = []
            self.history["train_loss"].append(loss.item())
        else:
            raise TypeError(
                f"Loss must be a dict or tensor, not {type(loss)}")

        return loss

    def train_epoch(self, epoch):

        self.model.train()

        if self.verbose:
            iterator = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        else:
            iterator = enumerate(self.dataloader)

        for b, batch_data in iterator:

            self.on_batch_start(b)

            if self.mode == 'vector_classification':

                img_inputs = batch_data[0]
                vector_inputs = batch_data[1]
                labels = batch_data[2]

                img_inputs = img_inputs.to(self.device)
                vector_inputs = vector_inputs.to(self.device)
                labels = labels.to(self.device)

                inputs = (img_inputs, vector_inputs)
            
            elif self.mode == 'heatmap_classification':
                inputs, labels = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            elif self.mode == 'regression':
                inputs, labels = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            else:
                raise ValueError(
                    f"Mode must be 'classification' or 'regression', not {self.mode}")

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)

            loss = self.handle_train_loss(loss)

            loss.backward()

            self.optimizer.step()

            self.on_batch_end(b)

    def validate_epoch(self, epoch):
        
        if self.mode == 'regression':
            inference_output = regression_inference(
                self.model, self.val_dataloader, self.device, self.verbose)
        elif self.mode == 'vector_classification':
            inference_output = vector_classification_inference(
                self.model, self.val_dataloader, self.device, self.verbose)
        elif self.mode == 'heatmap_classification':
            inference_output = heatmap_classification_inference(
                self.model, self.val_dataloader, self.device, self.verbose)
        else:
            raise ValueError(
                f"Mode must be 'classification' or 'regression', not {self.mode}")

        for metric in self.metrics:

            score = metric(*inference_output)
            if metric.__name__[:4] == 'val_':
                metric_name = metric.__name__
            else:
                metric_name = 'val_'+metric.__name__

            # Score may be dict or float
            if isinstance(score, dict):
                for key in score:
                    name = metric_name+"_"+key
                    if name not in self.history:
                        self.history[name] = []
                    self.history[name].append(score[key])
            else:
                if metric_name not in self.history:
                    self.history[metric_name] = []
                self.history[metric_name].append(score)

            if self.verbose:
                print(f"{metric_name}: {score}")

    def train(self):

        for epoch in range(self.num_epochs):

            self.on_epoch_start(epoch)
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.on_epoch_end(epoch)

        return self.history
    
def vector_classification_inference(model, dataloader, device, verbose=True):
    """
    Inference on a PyTorch model.
    Args:
        model (torch.nn.Module): Model to be inferred.
        dataloader (torch.utils.data.DataLoader): Dataloader to infer on.
        device (torch.device): Device to run the model on.
        verbose (bool): Whether to print the current batch number.
    Returns:
        list: List of model outputs.
    """
    model.eval()

    with torch.no_grad():

        all_outputs, all_labels = [], []

        for val_img_inputs, val_vector_inputs, val_labels in tqdm(dataloader, disable=not verbose):

            val_img_inputs = val_img_inputs.to(device)
            val_vector_inputs = val_vector_inputs.to(device)
            val_labels = val_labels.to(device)

            outputs = model((val_img_inputs, val_vector_inputs))

            all_outputs.append(outputs.cpu())
            all_labels.append(val_labels.cpu())

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

    return all_outputs, all_labels

def heatmap_classification_inference(model, dataloader, device, verbose=True):
    """
    Inference on a PyTorch model.
    Args:
        model (torch.nn.Module): Model to be inferred.
        dataloader (torch.utils.data.DataLoader): Dataloader to infer on.
        device (torch.device): Device to run the model on.
        verbose (bool): Whether to print the current batch number.
    Returns:
        list: List of model outputs.
    """
    model.eval()

    with torch.no_grad():

        all_outputs, all_labels = [], []

        for val_inputs, val_labels in tqdm(dataloader, disable=not verbose):

            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_inputs)

            all_outputs.append(outputs.cpu())
            all_labels.append(val_labels.cpu())

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

    return all_outputs, all_labels

def regression_inference(model, dataloader, device, verbose=True):
    """
    Inference on a PyTorch model.
    Args:
        model (torch.nn.Module): Model to be inferred.
        dataloader (torch.utils.data.DataLoader): Dataloader to infer on.
        device (torch.device): Device to run the model on.
        verbose (bool): Whether to print the current batch number.
    Returns:
        list: List of model outputs.
    """
    model.eval()

    with torch.no_grad():

        all_outputs, all_labels, all_coords = [], [], []

        for val_inputs, val_labels, val_coords in tqdm(dataloader, disable=not verbose):
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_inputs)

            all_outputs.append(outputs.cpu())
            all_labels.append(val_labels.cpu())
            all_coords.append(val_coords)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_coords = [
            coords for sublist in all_coords for coords in sublist]
        
    return all_outputs, all_labels, all_coords

class GeoModelTrainer:
    def __init__(self,
                 model,
                 epochs,
                 batch_size,
                 optimizer,
                 lrscheduler,
                 train_dataset,
                 val_dataset=None,
                 device=torch.device('cuda')):
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.scheduler = lrscheduler
        self.optimizer = optimizer
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}

    def criterion(self, y_pred, y_true):

        preds = []
        gts = []
        for key in y_pred:
            preds.append(y_pred[key])
            gts.append(y_true[key])

        preds = torch.cat(preds, dim=1)
        gts = torch.cat(gts, dim=1)
        
        loss = F.mse_loss(preds, gts, reduction='mean')

        return loss
    
    def train(self, verbose=True):
        for epoch in tqdm(range(self.epochs), disable=verbose):
            self.model.train()
            for x_batch, y_batch in tqdm(self.train_loader, disable=not verbose):
                # x_batch and y_batch are dictionaries
                x_batch = {key: value.to(self.device) for key, value in x_batch.items()}
                y_batch = {key: value.to(self.device) for key, value in y_batch.items()}
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.history['train_loss'].append(loss.item())
            if self.val_loader:
                self.validate()
            if verbose:
                clear_output(wait=True)
                print(f'Epoch {epoch + 1}/{self.epochs}')
                # Also show the learning rate
                plot_metric(self.history, ('train_loss', 'val_loss'))
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = {key: value.to(self.device) for key, value in x_batch.items()}
                y_batch = {key: value.to(self.device) for key, value in y_batch.items()}
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)