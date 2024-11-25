import torch
from torch import nn
from torch_geometric.data import Data
import pytorch_lightning as pl

from timm.scheduler import create_scheduler

import argparse
from typing import Optional, Dict, Any

from .model import SE3Set
from .optims import create_optimizer
from .loss import L2MAELoss


class Lit_SE3Set(pl.LightningModule):
    def __init__(self, 
                 model_params: Dict[str, Any], 
                 train_params: Dict[str, Any], 
                 dataset_statistics: Optional[Dict[str, Any]] = None,
                 task_mean: Optional[torch.Tensor] = None,
                 task_std: Optional[torch.Tensor] = None):
        super(Lit_SE3Set, self).__init__()
        self.save_hyperparameters()

        self.model = SE3Set(**model_params, dataset_statistics=dataset_statistics)

        self.standardize = train_params['standardize']
        if self.standardize:
            assert task_mean is not None and task_std is not None, "task_mean and task_std must be provided when standardize is True"
            self.task_mean = task_mean
            self.task_std = task_std
        
        self.train_params = train_params

        if train_params['criterion'].lower() in ['l1', 'mae']:
            self.criterion = nn.L1Loss()
        elif train_params['criterion'].lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif train_params['criterion'].lower() == 'huber':
            self.criterion = nn.HuberLoss(delta=train_params['huber_delta'])
        elif train_params['criterion'].lower() == 'l2mae':
            self.criterion = L2MAELoss()

        self.evaluation = nn.L1Loss()

        if self.model.output_force:
            self.weights = train_params['weights']

    def forward(self, data: Data):
        return self.model(data)

    def configure_optimizers(self):
        args = argparse.Namespace(**self.train_params)
        optimizer = create_optimizer(args, self.model)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        return {'optimizer': optimizer, 
                'lr_scheduler': {'scheduler': lr_scheduler, 
                                 'interval': 'epoch'}}
    
    def lr_scheduler_step(self, scheduler, metric):
        """Refer to https://github.com/Lightning-AI/pytorch-lightning/issues/5555#issuecomment-1065894281
        """
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def training_step(self, train_batch, batch_idx):
        output, force = self(train_batch)
        
        device = output.device

        if self.standardize:
            loss = self.criterion(output, (train_batch.y - self.task_mean.to(device)) / self.task_std.to(device))
        else:
            loss = self.criterion(output, train_batch.y)

        if self.model.output_force:
            if self.standardize:
                loss = loss * self.weights[0] + self.criterion(force, train_batch.force / self.task_std.to(device)) * self.weights[1]
            else:
                loss = loss * self.weights[0] + self.criterion(force, train_batch.force) * self.weights[1]

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.num_graphs)
        return loss
    
    def validation_step(self, val_batch, val_batch_idx):
        with torch.set_grad_enabled(self.model.output_force):
            val_output, val_force = self(val_batch)

        device = val_output.device

        if self.standardize:
            val_loss = self.evaluation(val_output * self.task_std.to(device) + self.task_mean.to(device), val_batch.y)
        else:
            val_loss = self.evaluation(val_output, val_batch.y)

        if self.model.output_force:
            val_energy_loss = val_loss
            if self.standardize:
                val_force_loss = self.evaluation(val_force * self.task_std.to(device), val_batch.force)
            else:
                val_force_loss = self.evaluation(val_force, val_batch.force)
                
            val_loss = self.weights[0] * val_energy_loss + self.weights[1] * val_force_loss

            self.log('val_energy_loss', val_energy_loss, batch_size=val_batch.num_graphs)
            self.log('val_forces_loss', val_force_loss, batch_size=val_batch.num_graphs)

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=val_batch.num_graphs)

    def test_step(self, test_batch, test_batch_idx):
        with torch.set_grad_enabled(self.model.output_force):
            test_output, test_force = self(test_batch)
        
        device = test_output.device

        if self.standardize:
            test_loss = self.evaluation(test_output * self.task_std.to(device) + self.task_mean.to(device), test_batch.y)
        else:
            test_loss = self.evaluation(test_output, test_batch.y)

        if self.model.output_force:
            test_energy_loss = test_loss
            if self.standardize:
                test_force_loss = self.evaluation(test_force * self.task_std.to(device), test_batch.force)
            else:
                test_force_loss = self.evaluation(test_force, test_batch.force)
                
            test_loss = self.weights[0] * test_energy_loss + self.weights[1] * test_force_loss

            self.log('test_energy_loss', test_energy_loss, batch_size=test_batch.num_graphs)
            self.log('test_forces_loss', test_force_loss, batch_size=test_batch.num_graphs)

        self.log('test_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=test_batch.num_graphs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.set_grad_enabled(self.model.output_force):
            output, force = self(batch)

        device = output.device

        if self.standardize:
            if self.model.output_force:
                force = force * self.task_std.to(device)
                
            return output * self.task_std.to(device) + self.task_mean.to(device), force
        else:
            return output, force
