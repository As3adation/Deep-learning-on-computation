import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader

from cs236781.train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        post_epoch_fn=None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # Train & evaluate for one epoch
            train_result = self.train_epoch(dl_train, **kw)
            test_result = self.test_epoch(dl_test, **kw)
            
            # Save results
            train_loss.extend(train_result.losses)
            train_acc.append(train_result.accuracy)
            test_loss.extend(test_result.losses)
            test_acc.append(test_result.accuracy)
            
            # Update best accuracy and early stopping counter
            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy
                epochs_without_improvement = 0
                save_checkpoint = True
            else:
                epochs_without_improvement += 1

            # Early stopping
            if early_stopping is not None and epochs_without_improvement >= early_stopping:
                break

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)
                
            actual_num_epochs = epoch + 1

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        # Initialize hidden state at the start of each epoch
        self.hidden_state = None    
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # Initialize hidden state at the start of each epoch
        self.hidden_state = None   
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # Forward pass
        y_pred, self.hidden_state = self.model(x, self.hidden_state)
        
        # Detach hidden state from graph while maintaining the tensor
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        
        # Reshape predictions and labels for loss calculation
        # y_pred is (B,S,V), reshape to (B*S,V)
        y_pred_reshaped = y_pred.view(-1, y_pred.shape[-1])
        # y is (B,S), reshape to (B*S)
        y_reshaped = y.view(-1)
        
        # Calculate loss
        loss = self.loss_fn(y_pred_reshaped, y_reshaped)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Calculate accuracy (number of correct predictions)
        predictions = torch.argmax(y_pred, dim=-1)  # (B,S)
        num_correct = (predictions == y).sum()

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            # Forward pass
            y_pred, self.hidden_state = self.model(x, self.hidden_state)
            
            # Reshape predictions and labels for loss calculation
            y_pred_reshaped = y_pred.view(-1, y_pred.shape[-1])
            y_reshaped = y.view(-1)
            
            # Calculate loss
            loss = self.loss_fn(y_pred_reshaped, y_reshaped)
            
            # Calculate accuracy
            predictions = torch.argmax(y_pred, dim=-1)
            num_correct = (predictions == y).sum()

        return BatchResult(loss.item(), num_correct.item() / seq_len)


class VAETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        # TODO: Train a VAE on one batch.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            # TODO: Evaluate a VAE on one batch.
            # ====== YOUR CODE: ======
            raise NotImplementedError()    
            # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())


class TransformerEncoderTrainer(Trainer):
    
    def train_batch(self, batch) -> BatchResult:
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].float().to(self.device)
        label = batch['label'].float().to(self.device)
        
        loss = None
        num_correct = None
        # TODO:
        #  fill out the training loop.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        
        
        
        return BatchResult(loss.item(), num_correct.item())
        
    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].float().to(self.device)
            label = batch['label'].float().to(self.device)
            
            loss = None
            num_correct = None
            
            # TODO:
            #  fill out the testing loop.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

            
        
        return BatchResult(loss.item(), num_correct.item())



class FineTuningTrainer(Trainer):
    
    def train_batch(self, batch) -> BatchResult:
        
        input_ids = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"]
        labels= batch["label"]
        # TODO:
        #  fill out the training loop.
        # ====== YOUR CODE: ======

        raise NotImplementedError()
        
        # ========================
        
        return BatchResult(loss, num_correct)
        
    def test_batch(self, batch) -> BatchResult:
        
        input_ids = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"]
        labels= batch["label"]
        
        with torch.no_grad():
            # TODO:
            #  fill out the training loop.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================
        return BatchResult(loss, num_correct)
