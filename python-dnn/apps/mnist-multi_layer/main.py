from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import heepstorch as hp

IMAGE_SIZE = 12


def get_model(device):
    # NOTE: We don't use Softmax in the model, as it's not necessary for training.
    #   Instead, we use the nn.CrossEntropyLoss function, which already performs
    #   the softmax inside. Therefore, if probabilities are desired on inference,
    #   the softmax function must be applied to the output of the model.
    HIDDEN_SIZE = 20

    model = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE)),
        ('relu0', nn.ReLU()),
        ('fc1', nn.Linear(HIDDEN_SIZE, 10)),
    ])).to(device)
    return model


def get_loaders(batch_size=64):
    class FlattenTransform:
        def __call__(self, img):
            return img.view(-1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        FlattenTransform()
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

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
                num_epochs=5, lr=0.01, checkpoint_path='mnist_model.pth'):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        accuracy = test(model, test_loader, criterion, device, f"epoch {epoch}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved with accuracy: {best_accuracy:.2f}%")
    return model


def load_or_train_model(train_loader, test_loader, criterion, device,
                        retrain=True, checkpoint_path='mnist_model.pth'):
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
    # Get first n_samples from loader
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

    for i, ax in enumerate(axes):
        ax.imshow(images[i].view(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        ax.set_title(
            f'True: {labels[i].item()}\nPred: {predictions[i].item()} (prob {probs[i, predictions[i]].item() * 100:.2f}%)')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Return matrices and predictions for example_main
    return {
        'input_matrix': images.numpy(),
        'expected_output_prob_matrix': probs.cpu().numpy(),
        'expected_predictions': predictions.cpu().numpy().tolist(),
        'true_label_values': labels.numpy().tolist()
    }


def main(retrain, use_gpu_if_available):
    device = torch.device('cuda' if use_gpu_if_available and torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    criterion = torch.nn.CrossEntropyLoss()

    model = load_or_train_model(train_loader, test_loader, criterion, device, retrain)

    hp_nn = hp.module.SequentialNetwork.from_torch_sequential(model)
    quantized_torch_model = hp_nn.get_quantized_torch_module()

    test(quantized_torch_model, test_loader, criterion, device, "quantized")
    test(model, test_loader, criterion, device, "non-quantized")

    pred_res = get_test_predictions(quantized_torch_model, test_loader, device, 5)
    # pprint.pprint(pred_res)

    cg = hp.code_generator.CodeGenerator('mnist-multi_layer', hp_nn)
    cg.generate_code(append_final_softmax=True, overwrite_existing_generated_files=True)
    cg.generate_example_main('main.cpp', pred_res['input_matrix'], pred_res['expected_output_prob_matrix'],
                             pred_res['expected_predictions'], pred_res['true_label_values'],
                             overwrite_existing_generated_files=True)


if __name__ == "__main__":
    main(retrain=False, use_gpu_if_available=True)
