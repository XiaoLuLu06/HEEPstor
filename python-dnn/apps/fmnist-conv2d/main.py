from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import heepstorch as hp
import pprint

IMAGE_SIZE = 14
CHANNELS = 1  # Fashion MNIST is grayscale


def get_model(device):
    model = nn.Sequential(OrderedDict([
        # First conv block
        ('conv1', nn.Conv2d(1, 16, kernel_size=4, padding=0)),
        ('relu1', nn.ReLU()),

        # Second conv block
        ('conv2', nn.Conv2d(16, 8, kernel_size=4, padding=0)),
        ('relu2', nn.ReLU()),

        # Flatten and output
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(8 * 8 * 8, 10))
    ])).to(device)
    return model


def get_loaders(batch_size=64):
    # Fashion MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, test_loader, criterion, device, description=""):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set {description}: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def train_model(model, train_loader, test_loader, criterion, device,
                num_epochs=10, lr=0.001, checkpoint_path='fashion_mnist_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        accuracy = test(model, test_loader, criterion, device, f"epoch {epoch}")
        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved with accuracy: {best_accuracy:.2f}%")
    return model


def load_or_train_model(train_loader, test_loader, criterion, device,
                        retrain=True, checkpoint_path='fashion_mnist_model.pth'):
    model = get_model(device)
    if os.path.exists(checkpoint_path) and not retrain:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
        return model

    return train_model(model, train_loader, test_loader, criterion, device,
                       checkpoint_path=checkpoint_path)


def get_test_predictions(model, test_loader, device, n_samples: int):
    """
    Gets first n_samples from test set and computes predictions and matrices
    """
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:n_samples]
    labels = labels[:n_samples]

    model.eval()
    with torch.no_grad():
        output = model(images.to(device))
        probs = torch.softmax(output, dim=1)
        predictions = probs.argmax(dim=1)

    # Create visualization
    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    if n_samples == 1:
        axes = [axes]

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for i, ax in enumerate(axes):
        # Reshape for display (single channel)
        img_display = images[i].squeeze()  # Remove channel dimension for grayscale
        ax.imshow(img_display, cmap='gray')
        ax.set_title(
            f'True: {classes[labels[i]]}\nPred: {classes[predictions[i]]} \n(prob {probs[i, predictions[i]].item() * 100:.2f}%)')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return {
        'input_matrix': images.numpy(),
        'expected_output_prob_matrix': probs.cpu().numpy(),
        'expected_predictions': predictions.cpu().numpy().tolist(),
        'true_label_values': labels.numpy().tolist()
    }


def print_test_predictions_forward_pass(test_loader, hp_nn: hp.module.SequentialNetwork, n_samples: int):
    """
       Gets first n_samples from test set and computes predictions and matrices
    """
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:n_samples]
    labels = labels[:n_samples]

    probs = []
    predictions = []

    for i in range(n_samples):
        image = hp.code_generator.flatten_input_to_matrix(images[i].detach().numpy())
        output = hp_nn(image)
        all_probs = torch.softmax(torch.from_numpy(output), 1)
        pred = int(torch.argmax(all_probs).item())

        predictions.append(pred)
        probs.append(all_probs[0, pred])

    print(f'Heepstorch forward pass predictions: {predictions}')

    # Create visualization
    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    if n_samples == 1:
        axes = [axes]

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for i, ax in enumerate(axes):
        # Reshape for display (single channel)
        img_display = images[i].squeeze()  # Remove channel dimension for grayscale
        ax.imshow(img_display, cmap='gray')
        ax.set_title(
            f'True: {classes[labels[i]]}\nPred: {classes[predictions[i]]} \n(prob {probs[i] * 100:.2f}%)')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main(retrain=False, use_gpu_if_available=True):
    device = torch.device('cuda' if use_gpu_if_available and torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    criterion = torch.nn.CrossEntropyLoss()

    model = load_or_train_model(train_loader, test_loader, criterion, device, retrain)
    # get_test_predictions(model, test_loader, device, 5)

    hp_nn = hp.module.SequentialNetwork.from_torch_sequential(model, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    quantized_torch_model = hp_nn.get_quantized_torch_module()

    test(quantized_torch_model, test_loader, criterion, device, "quantized")
    test(model, test_loader, criterion, device, "non-quantized")

    pred_res = get_test_predictions(quantized_torch_model, test_loader, device, 1)
    pprint.pprint(pred_res)

    print_test_predictions_forward_pass(test_loader, hp_nn, 1)

    cg = hp.code_generator.CodeGenerator('fmnist-conv2d', hp_nn)
    cg.generate_code(append_final_softmax=True, overwrite_existing_generated_files=True)
    cg.generate_example_main('main.cpp', hp.code_generator.flatten_input_to_matrix(pred_res['input_matrix']),
                             pred_res['expected_output_prob_matrix'],
                             pred_res['expected_predictions'], pred_res['true_label_values'],
                             overwrite_existing_generated_files=True)


if __name__ == "__main__":
    main(retrain=False, use_gpu_if_available=True)
