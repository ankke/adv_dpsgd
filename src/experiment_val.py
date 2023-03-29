import torch
import config
import time
import copy

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager

from adv import Attack
from utils import plot_and_save
from model import Model

class Experiment:
    def __init__(self, batch_size, epochs, adv_attack, adv_attack_mode, adv_params, device, id,
                save=True, dp=True, target_epsilon=7.5, dataset='cifar', adv_test=False, max_batch_size=512, max_grad_norm=1.5, dir_name=None):
        self.batch_size=batch_size
        self.epochs = epochs
        self.attack = Attack(adv_attack, adv_params, device) if adv_attack is not None else None
        self.adv_attack_mode = adv_attack_mode
        self.dp = dp
        self.device = device
        self.adv_test = adv_test
        self.max_batch_size = max_batch_size
        self.target_epsilon = target_epsilon
        self.dataset = dataset
        self.adv_test = adv_test
        self.max_batch_size = max_batch_size
        self.max_grad_norm = max_grad_norm
        self.save = save
        self.id = id
        self.best_epoch = epochs
        self.best_weights = None
        self.best_val = 0.0

        self.dir_name = f'results_{self.dataset}' if dir_name is None else dir_name
        self.criterion = nn.CrossEntropyLoss() 
        self.model = Model(dataset) 
        self.optimizer, self.train_loader, self.test_loader = self.model.setup(epochs, batch_size, dp, target_epsilon, max_grad_norm, device)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

    def _log(self, message):
        if config.verbose:
            print(f"Experiment {self.id}: {message}")

    def _fit(self):
        self.model.to(self.device)
        self._log("Training started")
        train_losses = []
        train_accuracies = []

        val_losses = []
        val_accuracies = []

        best_acc = 0.0

        for epoch in range(self.epochs):
            if self.dp:
                with BatchMemoryManager(
                        data_loader=self.train_loader,
                        max_physical_batch_size=self.max_batch_size,
                        optimizer=self.optimizer
                ) as new_train_loader:
                    train_loss, train_acc = self._run_epoch(new_train_loader)
            else:
                train_loss, train_acc = self._run_epoch(self.train_loader)

            self._log(f"Epoch {epoch + 1: >3}/{self.epochs}, train loss: {train_loss:.2e}, train acc: {train_acc:.3f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            self.lr_scheduler.step(train_loss)
            

            val_loss, val_acc = self._run_epoch(self.test_loader, eval=True)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            # self.lr_scheduler.step(val_loss)
            
            self._log(f"Epoch {epoch + 1: >3}/{self.epochs}, val loss: {val_loss:.2e}, val acc: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                self.best_val = best_acc
                self.best_weights = copy.deepcopy(self.model.net.state_dict())
                self.best_epoch = epoch + 1

        plot_and_save(range(len(train_losses)), np.array(train_losses), 'train_loss', f'{self.dir_name}/{self.id}', self.save)
        plot_and_save(range(len(train_accuracies)), np.array(train_accuracies), 'train_acc', f'{self.dir_name}/{self.id}', self.save)
        plot_and_save(range(len(val_losses)), np.array(val_losses), 'val_loss', f'{self.dir_name}/{self.id}', self.save)
        plot_and_save(range(len(val_accuracies)), np.array(val_accuracies), 'val_acc', f'{self.dir_name}/{self.id}', self.save)

    def _run_epoch(self, data_loader, eval=False):
        self.model.mode(eval)

        epoch_losses = []
        epoch_acc = []

        for inputs, targets in tqdm(data_loader, disable=config.disable_tqdm):
            if self.attack is None or eval:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:
                inputs, targets = self.attack.perturbe(inputs, targets, self.model.get_model(self.dp), self.adv_attack_mode)

            self.optimizer.zero_grad()

            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, targets)

            top1 = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = targets.detach().cpu().numpy()
            acc = (top1 == labels).mean()
            epoch_losses.append(loss.item())
            epoch_acc.append(acc)

            if not eval:
                loss.backward()
                self.optimizer.step()

        epoch_loss = np.mean(epoch_losses)
        epoch_acc = np.mean(epoch_acc)

        return epoch_loss, epoch_acc
    
    def repr_(self):
        adv = self.attack.attack.__name__ if self.attack is not None else 'non adv'
        adv_params = str(self.attack.params) if self.attack is not None else 'non_adv'
        params = f'{self.id},{self.batch_size},{self.epochs},{adv},{self.adv_attack_mode},{adv_params},{self.dp},{self.target_epsilon},{self.dataset},{self.max_batch_size},{self.max_grad_norm}'
        return params

    def run(self):
        torch.cuda.empty_cache()
        repr = self.repr_()
        print(repr)

        start_time = time.time()
        self._fit()
        end_time = time.time()
        trainig_time = end_time - start_time

        # self.model.net.load_state_dict(self.best_weights)
        print(f'best val acc: {self.best_val}, epoch: {self.best_epoch}')

        _, test_acc = self._run_epoch(self.test_loader, eval=True)
        _, train_acc = self._run_epoch(self.train_loader, eval=True)

        print(f"Test accuracy: {test_acc}")
        print("Training time: ", time.strftime("%M:%S", time.gmtime(trainig_time)))
        
        if self.save:
            with open(f'{self.dir_name}/history.csv', 'a') as fd:
                fd.write(f'\n{repr},{trainig_time},{train_acc},{test_acc}')


class Experiment_weighted(Experiment):
    def __init__(self, batch_size, epochs, adv_attack, adv_attack_mode, adv_params, device, id,
                save=True, dp=True, target_epsilon=7.5, dataset='cifar', adv_test=False, max_batch_size=512, max_grad_norm=1.5, clean_weight=9, adv_weight=1, dir_name=None):
        dir_name = f'results_{dataset}_weighted' if dir_name is None else dir_name
        super().__init__(batch_size, epochs, adv_attack, adv_attack_mode, adv_params, device, id,
                save, dp, target_epsilon, dataset, adv_test, max_batch_size, max_grad_norm, dir_name)
        self.clean_weight = clean_weight
        self.adv_weight = adv_weight
    
    def repr_(self):
        adv = self.attack.attack.__name__ if self.attack is not None else 'non adv'
        adv_params = str(self.attack.params) if self.attack is not None else 'non_adv'
        params = f'{self.id},{self.batch_size},{self.epochs},{adv},{self.adv_attack_mode},{adv_params},{self.dp},{self.target_epsilon},{self.dataset},{self.max_batch_size},{self.max_grad_norm},{self.clean_weight},{self.adv_weight}'
        return params

    def _run_epoch(self, data_loader, eval=False):
        if eval:
            return super()._run_epoch(data_loader, eval)
        else:
            return self._run_epoch_weighted(data_loader)
    
    def _run_epoch_weighted(self, data_loader):

        self.model.mode(eval=False)


        epoch_losses = []
        epoch_acc = []

        for inputs, targets in tqdm(data_loader, disable=config.disable_tqdm):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_adv, _ = self.attack.perturbe(inputs, targets, self.model.get_model(self.dp), self.adv_attack_mode)

            self.optimizer.zero_grad()

            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, targets)

            outputs_adv = self.model.forward(inputs_adv)
            loss_adv = self.criterion(outputs_adv, targets)
            
            loss_weighted = (loss * self.clean_weight + loss_adv * self.adv_weight) / (self.clean_weight + self.adv_weight)

            top1 = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = targets.detach().cpu().numpy()
            acc = (top1 == labels).mean()
            epoch_losses.append(loss_weighted.item())
            epoch_acc.append(acc)

            loss_weighted.backward()
            self.optimizer.step()

        epoch_loss = np.mean(epoch_losses)
        epoch_acc = np.mean(epoch_acc)

        return epoch_loss, epoch_acc
    