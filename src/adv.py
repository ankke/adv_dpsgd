import torch
import matplotlib.pyplot as plt

import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class Attack:
    def __init__(self, attack, params, device):
        self.attack = attack
        self.params = params
        self.device = device

    def perturbe(self, inputs, targets, model, mode):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if mode == 'batch':
            return self.attack_per_batch(inputs, targets, model)
        elif mode == 'sample':
            return self.attack_per_sample(inputs, targets, model)

    def attack_per_batch(self, inputs, targets, model):
        attack = self.attack(model, **self.params)

        perturbed_inputs = attack(inputs, targets)
        
    #         plt.imshow(np.transpose(torchvision.utils.make_grid(perturbed_inputs).cpu().numpy(), (1, 2, 0)))
    #         plt.show()
    #         plt.imshow(np.transpose(torchvision.utils.make_grid(inputs).cpu().numpy(), (1, 2, 0)))
    #         plt.show()
        
        return perturbed_inputs, targets


    def attack_per_sample(self, inputs, targets, model):
        adv_images = torch.empty_like(inputs)
        attack = self.attack(model, **self.params)
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

        adv_images.to(self.device)
        # print(adv_images.shape, inputs.shape)
        return adv_images, targets


def adv_test(model, dataloader, classes, device, attack):
    print("Adversarial robustness test started")
    accuracies = []
    examples = []
    epsilons = [0, .05, .1]

    for eps in epsilons:
        params = {'eps': eps}
        acc, ex = test(eps, model, dataloader, device, attack.attack, params)
        accuracies.append(acc)
        examples.append(ex)
    
    plot(epsilons, accuracies, examples, classes)


def test(eps, model, data_loader, device, attack, params):
        correct = 0
        adv_examples = []
   
        model.eval()

        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            init_pred = torch.argmax(output, dim=1).long()

            if init_pred.item() != target.item():
                continue
            print(len(data))
            perturbed_inputs = attack(model, **params)
            output = model(perturbed_inputs)
            print(3)

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
        print(
            "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(data_loader), final_acc))
        return final_acc, adv_examples



def plot(epsilons, accuracies, examples, classes):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, max(accuracies) + 0.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

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
            plt.title("{} -> {}".format(classes[orig], classes[adv]))
            plt.imshow(np.transpose(ex, (1, 2, 0)))
    plt.tight_layout()
    plt.show()