import torch
import os
import opacus
import copy
import torchvision
import torchattacks

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from model import ResNet9, initialize_weights

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

class Experiment:
    def __init__(self, batch_size, epochs, patience, adv_attack, adv_attack_mode, epsilon, dp, device, save_experiment,
                 verbose, dataset='cifar', adv_test=True, name=None):
        self.model = ResNet9(norm_layer="group")
        self.adv_model = ResNet9(norm_layer="group").to(device)
        self.optimizer = optim.NAdam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.adv_attack = adv_attack
        self.adv_attack_mode = adv_attack_mode
        self.epsilon = epsilon
        self.dp = dp
        self.device = device
        self.verbose = verbose
        self.disable_tqdm = not self.verbose
        self.save_experiment = save_experiment
        self.dataset = dataset
        self.adv_test = adv_test
        self.best_model_weights = None


        if name is None:
            adv_s = f"adv-{epsilon}-{adv_attack_mode}" if adv_attack is not None else "non_adv"
            dp_s = "dp" if dp else "non_dp"
            self.name = f"{dataset}+{adv_s}+{dp_s}+{batch_size}"
        else:
            self.name = name

        now = datetime.now()
        formatted_timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
        self.dir_name = f"results/{self.name}_{formatted_timestamp}"
        if self.save_experiment:
            os.makedirs(self.dir_name, exist_ok=True)

        self._setup_training()

    def _log(self, message):
        print(f"Experiment {self.name}: {message}")

    def _setup_training(self):
        self._log("Loading data")
        if self.dataset == 'cifar':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
                ])
            learning_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

            train_set, val_set = torch.utils.data.random_split(learning_set, [35000, 15000],
                                                            generator=torch.Generator().manual_seed(42))
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                ])
            learning_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            train_set, val_set = torch.utils.data.random_split(learning_set, [40000, 20000],
                                                           generator=torch.Generator().manual_seed(42))
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            self.classes = (x for x in range(0,9))

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, num_workers=2)
        self.val_loader_x1 = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


        if self.dp:
            self._log("DP on")
            self.privacy_engine = opacus.PrivacyEngine()
            # self.model = ModuleValidator.fix(self.model)
            # self.optimizer = optim.NAdam(self.model.parameters())
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
            )
        else:
            self._log("DP off")

    def _run_epoch(self, data_loader):
        self.model.train()

        dataset_len = len(data_loader.dataset)

        epoch_loss = 0.0
        epoch_acc = 0.0

        for inputs, targets in tqdm(data_loader, disable=self.disable_tqdm):
            if self.adv_attack:
                if self.adv_attack_mode == 'batch':
                    inputs, targets = self._fgsm_attack_batch(inputs, targets)
                else:
                    inputs, targets = self._fgsm_attack_per_sample(inputs, targets)
            else:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            top1 = torch.argmax(outputs, dim=1).long()
            n_correct = torch.sum(top1 == targets)
            epoch_loss += loss.item()
            epoch_acc += n_correct.item()

            loss.backward()
            self.optimizer.step()

        epoch_loss /= dataset_len
        epoch_acc /= dataset_len

        return epoch_loss, epoch_acc

    def _fgsm_attack_per_sample(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        adv_images = torch.empty_like(inputs)
        if self.dp:
            self.adv_model.load_state_dict(copy.deepcopy(self.model._module.state_dict()))
            attack = torchattacks.FGSM(self.adv_model, eps=self.epsilon)
        else:
            attack = torchattacks.FGSM(self.model, eps=self.epsilon)
        for i in range(len(inputs)):
            sample = torch.unsqueeze(inputs[i], dim=0)
            sample_target = targets[i:i + 1]

            # plt.imshow(np.transpose(sample.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(adv_image.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            # plt.show()
            adv_images[i] = attack(sample, sample_target)

        # imshow(torchvision.utils.make_grid(adv_images).cpu())
        # imshow(torchvision.utils.make_grid(inputs).cpu())
        # print(' '.join(f'{classes[targets[j]]:5s}' for j in range(BATCH_SIZE)))

        adv_images.to(self.device)
        # print(adv_images.shape, inputs.shape)
        return adv_images, targets

    def _fgsm_attack_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if self.dp:
            self.adv_model.load_state_dict(copy.deepcopy(self.model._module.state_dict()))
            attack = torchattacks.FGSM(self.adv_model, eps=self.epsilon)
        else:
            attack = torchattacks.FGSM(self.model, eps=self.epsilon)
    
        perturbed_inputs = attack(inputs, targets)
        
#         plt.imshow(np.transpose(torchvision.utils.make_grid(perturbed_inputs).cpu().numpy(), (1, 2, 0)))
#         plt.show()
#         plt.imshow(np.transpose(torchvision.utils.make_grid(inputs).cpu().numpy(), (1, 2, 0)))
#         plt.show()
#         print(' '.join(f'{self.classes[targets[j]]:5s}' for j in range(self.batch_size)))
        
        return perturbed_inputs, targets

    def _fit(self):
        self._log("Training started")
        _ = self.model.to(self.device)
        _ = self.model.apply(initialize_weights)

        curr_patience = 0
        max_val_acc = 0

        val_losses = []
        val_accuracies = []
        train_losses = []
        train_accuracies = []

        for epoch in range(self.epochs):

            train_loss, train_acc = self._run_epoch(self.train_loader)
            self._log(f"Epoch {epoch + 1: >3}/{self.epochs}, train loss: {train_loss:.2e}, train acc: {train_acc:.3f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc, val_top5 = self._validate(self.val_loader)
            self._log(
                f"Epoch {epoch + 1: >3}/{self.epochs}, val loss: {val_loss:.2e}, val acc: {val_acc:.3f}, val top5: {val_top5:.3f}")

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if max_val_acc >= val_acc:
                curr_patience += 1
                if curr_patience == self.patience:
                    break
            else:
                curr_patience = 0
                max_val_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                if self.save_experiment:
                    torch.save(self.model.state_dict(), f"{self.dir_name}/{self.name}.pt")

        self.model.load_state_dict(self.best_model_weights)

        epsilon = self.privacy_engine.get_epsilon(1e-5)
        print(epsilon)
        self._log("Training finished")

        # plt.plot(range(len(val_losses)), np.array(val_losses))
        # plt.title('val loss')
        # if self.save_experiment:
        #     plt.savefig(f"{self.dir_name}/val_loss.png")
        # if self.verbose:
        #     plt.show()
        # else:
        #     plt.clf()

        # plt.plot(range(len(val_accuracies)), np.array(val_accuracies))
        # plt.title('val acc')
        # if self.save_experiment:
        #     plt.savefig(f"{self.dir_name}/val_acc.png")
        # if self.verbose:
        #     plt.show()
        # else:
        #     plt.clf()

        plt.plot(range(len(train_losses)), np.array(train_losses))
        plt.title('train loss')
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/train_loss.png")
        if self.verbose:
            plt.show()
        else:
            plt.clf()

        plt.plot(range(len(train_accuracies)), np.array(train_accuracies))
        plt.title('train acc')
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/train_acc.png")
        if self.verbose:
            plt.show()
        else:
            plt.clf()

    def _validate(self, data_loader):
        self.model.eval()
        epoch_acc = 0.0
        epoch_acc_top5 = 0.0
        epoch_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                top1 = torch.argmax(outputs, dim=1).long()
                n_correct = torch.sum(top1 == targets)

                _, top5 = torch.topk(outputs, 5, dim=1)
                top5 = top5.t()
                correct = top5.eq(targets.reshape(1, -1).expand_as(top5))
                n_correct_top5 = correct[:5].reshape(-1).float().sum(0, keepdim=True)

                epoch_loss += loss.item()
                epoch_acc += n_correct.item()
                epoch_acc_top5 += n_correct_top5.item()

            epoch_loss /= len(data_loader.dataset)
            epoch_acc /= len(data_loader.dataset)
            epoch_acc_top5 /= len(data_loader.dataset)

        return epoch_loss, epoch_acc, epoch_acc_top5

    def test(self, eps, data_loader):
        correct = 0
        adv_examples = []
        self.model.eval()

        if self.dp:
            self.adv_model.load_state_dict(copy.deepcopy(self.model._module.state_dict()))
            attack = torchattacks.FGSM(self.adv_model, eps=eps)
        else:
            attack = torchattacks.FGSM(self.model, eps=eps)

        for data, target in tqdm(data_loader, disable=self.disable_tqdm):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            init_pred = torch.argmax(output, dim=1).long()

            if init_pred.item() != target.item():
                continue

            perturbed_inputs = attack(data, target)
            output = self.model(perturbed_inputs)

            final_pred = torch.argmax(output, dim=1).long()
            if final_pred.item() == target.item():
                correct += 1

                if (eps == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_inputs.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_inputs.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        final_acc = correct / float(len(data_loader))
        self._log(
            "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(data_loader), final_acc))
        return final_acc, adv_examples

    def run(self):
        self._fit()
        self._log(f"Val accuracy: {self._validate(self.val_loader_x1)}")
        if not self.adv_test:
            return

        self._log("Adversarial robustness test started")
        accuracies = []
        examples = []
        epsilons = [0, .1, .2, .3]

        for eps in epsilons:
            acc, ex = self.test(eps, self.val_loader_x1)
            accuracies.append(acc)
            examples.append(ex)
        plt.figure(figsize=(5, 5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, max(accuracies) + 0.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/acc_vs_eps.png")
        if self.verbose:
            plt.show()
        else:
            plt.clf()

        cnt = 0
        plt.figure(figsize=(8, 10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons), len(examples[0]), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig, adv, ex = examples[i][j]
                plt.title("{} -> {}".format(self.classes[orig], self.classes[adv]))
                plt.imshow(np.transpose(ex, (1, 2, 0)))
        plt.tight_layout()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/perturbed.png")
        if self.verbose:
            plt.show()
        else:
            plt.clf()
