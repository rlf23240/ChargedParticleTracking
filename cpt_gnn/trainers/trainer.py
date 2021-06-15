#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable
from time import time
from pathlib import Path

import torch
import torch.optim as optimizers
import torch.nn as nn


class Trainer:
    """
    PyTorch model trainer.

    Implement common training logic.
    Override **train_epoch** and **evaluate**
    to perform custom training if needed.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        optimizer: optimizers.Optimizer,
        save: Path = None
    ):
        """
        :param model: A PyTorch nn module.
        :param loss: A PyTorch loss function.
        :param optimizer: A PyTorch optimizer instance.
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.save = save

        # self.device = 'cpu'

    def summary(self):
        """
        Output summary.
        """
        print("Model: \n{model}\nParameters: {params}".format(
            model=self.model,
            params=[param for param in self.model.parameters()]
        ))

    def write_checkpoint(self, checkpoint_id):
        """
        Write a checkpoint for the model.
        """
        if self.save is not None:
            checkpoint_dir = self.save / 'checkpoints'
            checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                obj={'model': self.model.state_dict()},
                f=checkpoint_dir / checkpoint_file
            )

    def train_epoch(self, data):
        """
        Train one epoch.
        Override if you want to perform custom training.

        :param data: Input data.
        :return: None
        """
        # Mark model for training.
        self.model.train()

        # Number of batches.
        n_batch = len(data)

        # Record train loss.
        train_loss = 0

        # Loss weight.
        real_weight = 1.0
        fake_weight = 1.0

        # Timing.
        start_time = time()

        # Train batches.
        for batch_idx, (batch_input, batch_target) in enumerate(data):
            print(f"Process batch {batch_idx}...")

            # Assign device.
            # batch_input = [a.to(self.device) for a in batch_input]
            # batch_target = batch_target.to(self.device)

            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch_target * real_weight
            batch_weights_fake = (1 - batch_target) * fake_weight
            batch_weights = batch_weights_real + batch_weights_fake

            # Reset gradient.
            self.model.zero_grad()

            # Predictions.
            batch_output = self.model(
                batch_input
            )

            # Loss function.
            batch_loss = self.loss(
                batch_output,
                batch_target,
                weight=batch_weights
            )

            # Backward propagation.
            batch_loss.backward()
            self.optimizer.step()

            # Record loss.
            loss = batch_loss.item()
            train_loss += loss

            print(f'batch {batch_idx}, loss {loss}')

        train_time = time() - start_time
        train_loss = train_loss/n_batch

        print(f'Processed {n_batch} batches in {train_time}s')
        print(f'Training loss: {train_loss}')

    @torch.no_grad()
    def evaluate(self, data):
        """
        Evaluate model.

        :param data: Input data.
        :return: None
        """
        # Mark model for evaluations.
        self.model.eval()

        # Number of batches.
        n_batch = len(data)

        # ACC parameters.
        correct_predictions = 0
        total_predictions = 0

        # Loss.
        valid_loss = 0

        # Timing.
        start_time = time()

        # Loop over batches
        for batch_idx, (batch_input, batch_target) in enumerate(data):
            # Assign to device.
            # batch_input = [a.to(self.device) for a in batch_input]
            # batch_target = batch_target.to(self.device)

            # Predictions.
            batch_output = self.model(batch_input)

            # Evaluate loss.
            loss = self.loss(batch_output, batch_target).item()
            valid_loss += loss

            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            correct_predictions += matches.sum().item()
            total_predictions += matches.numel()

        valid_time = time() - start_time
        valid_loss = valid_loss / n_batch
        valid_acc = correct_predictions / total_predictions

        print(
            f'Processed {total_predictions} samples '
            f'in {n_batch} batches '
            f'in {valid_time}s.'
        )
        print(
            f'Validation loss: {valid_loss} acc: {valid_acc}'
        )

    def train(self, train_data, epochs, valid_data=None):
        """
        Start training.

        :param train_data: Data use for training.
        :param epochs: Train epochs.
        :param valid_data: Data use for validation.
        """
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            # Train one epoch.
            self.train_epoch(train_data)
            self.write_checkpoint(epoch)
            # Validations.
            if valid_data is not None:
                self.evaluate(valid_data)

        if self.save is not None:
            torch.save(self.model.state_dict(), self.save / 'model')

