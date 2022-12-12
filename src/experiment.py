import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime
from .model import Net
import os
import opacus


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class Experiment:
    def __init__(self, batch_size, epochs, patience, adv_attack, adv_attack_mode, epsilon, dp, device, name=None, save_experiment=False, verbose=True):
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model = Net().cuda() if torch.cuda.is_available() else Net()
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
        self.best_model_weights = None

        if name is None:
            adv_s = f"adv-{epsilon}" if adv_attack is not None else "non_adv"
            dp_s = "dp" if dp else "non_dp"
            self.name = f"{adv_s}+{dp_s}+{batch_size}"
        else:
            self.name = name

        now = datetime.now()
        formatted_timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
        self.dir_name = f"{self.name}_{formatted_timestamp}"
        if self.save_experiment:
            os.mkdir(self.dir_name)

        self._setup_training()

    def _log(self, message):
        if self.verbose:
            print(f"Experiment {self.name}: {message}")

    def _setup_training(self):
        self._log("Loading data")
        transform = transforms.Compose(
            [transforms.ToTensor(),
             #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
             ])
        learning_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        train_set, val_set = torch.utils.data.random_split(learning_set, [35000, 15000], generator=torch.Generator().manual_seed(42))

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, num_workers=2)
        self.val_loader_x1 = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if self.dp:
            self.privacy_engine = opacus.PrivacyEngine()
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
            )


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
            self.model.zero_grad()

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

        self.model.eval()

        for i in range(len(inputs)):
            sample = torch.unsqueeze(inputs[i], dim=0)
            sample_target = targets[i:i + 1]

            sample.requires_grad = True
            self.model.zero_grad()

            output = self.model(sample)
            loss = self.criterion(output, sample_target)

            loss.backward()
            grad = sample.grad.data

            adv_image = sample + self.epsilon * grad.sign()
            adv_image = torch.clamp(adv_image, min=0, max=1)

            # plt.imshow(np.transpose(sample.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(adv_image.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            # plt.show()
            adv_images[i] = adv_image

        # imshow(torchvision.utils.make_grid(adv_images).cpu())
        # imshow(torchvision.utils.make_grid(inputs).cpu())
        # print(' '.join(f'{classes[targets[j]]:5s}' for j in range(BATCH_SIZE)))

        adv_images.to(self.device)
        # print(adv_images.shape, inputs.shape)
        self.model.train()
        return adv_images, targets

    def _fgsm_attack_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        inputs.requires_grad = True
        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        grad = inputs.grad.data

        adv_images = inputs + self.epsilon * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1)

        return adv_images, targets

    def _fit(self):
        self._log("Training started")
        _ = self.model.to(self.device)
        _ = self.model.apply(initialize_weights)

        curr_patience = 0
        min_val_loss = 1.0

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

            if val_loss >= min_val_loss:
                curr_patience += 1
                if curr_patience == self.patience:
                    break
            else:
                curr_patience = 0
                min_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                if self.save_experiment:
                    torch.save(self.model.state_dict(), f"{self.name}.pt")

        self.model.load_state_dict(self.best_model_weights)

        self._log("Training finished")

        plt.plot(range(len(val_losses)), np.array(val_losses))
        plt.title('val loss')
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/val_loss.png")

        plt.plot(range(len(val_accuracies)), np.array(val_accuracies))
        plt.title('val acc')
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/val_acc.png")

        plt.plot(range(len(train_losses)), np.array(train_losses))
        plt.title('train loss')
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/train_loss.png")

        plt.plot(range(len(train_accuracies)), np.array(train_accuracies))
        plt.title('train acc')
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/train_acc.png")

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

    def fgsm_attack(self, image, eps, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + eps * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def test(self, eps, data_loader):
        correct = 0
        adv_examples = []
        self.model.eval()

        for data, target in tqdm(data_loader, disable=self.disable_tqdm):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            self.model.zero_grad()
            output = self.model(data)
            init_pred = torch.argmax(output, dim=1).long()

            if init_pred.item() != target.item():
                continue

            loss = self.criterion(output, target)
            loss.backward()

            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, eps, data_grad)

            output = self.model(perturbed_data)

            final_pred = torch.argmax(output, dim=1).long()
            if final_pred.item() == target.item():
                correct += 1

                if (self.epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        final_acc = correct / float(len(data_loader))
        self._log("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(self.epsilon, correct, len(data_loader), final_acc))
        return final_acc, adv_examples

    def run(self, save_experiment=False, verbose=True):
        self.save_experiment = save_experiment
        self.verbose = verbose
        self._fit()
        self._log(f"Val accuracy: {self._validate(self.val_loader_x1)}")

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
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/acc_vs_eps.png")

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
        if self.verbose:
            plt.show()
        if self.save_experiment:
            plt.savefig(f"{self.dir_name}/perturbed.png")


