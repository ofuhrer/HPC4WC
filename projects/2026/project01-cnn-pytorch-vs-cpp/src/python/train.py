import os
import json
import torch
import argparse
import numpy as np
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision import transforms

from sklearn.model_selection import KFold

from model import CNN, reset_weights


# Import dataset. download=True is idempotent: torchvision skips the download
# when the raw idx files are already present (e.g. fetched by download_data.py
# or a previous run), so this only hits the network on a first local run.
_data_root = os.path.join(os.path.dirname(__file__), '../../data')
dataset_train_part = MNIST(
    _data_root, download=True, transform=transforms.ToTensor(), train=True)
dataset_test_part = MNIST(
    _data_root, download=True, transform=transforms.ToTensor(), train=False)


# Training parameters
K_FOLDS = 5
NUM_EPOCHS = 5
BATCH_SIZE = 10

# Fixed
SEED = 0

# Set number of threads for CPU to 1
torch.set_num_threads(1)


def train_model(network, trainloader, optimizer, loss_function, num_epochs, device):
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        current_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()        
            outputs = network(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                current_loss = 0.0


def test_model(network, testloader, device):
   # Function to test the model on the test set for a fold
    correct, total = 0, 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


def k_fold_cross_validation(k_folds, num_epochs, loss_function, batch_size, device):
    # Function to perform k-fold cross-validation
    results = {}
    best_accuracy = -1
    best_network = None
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        trainloader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler)

        network = CNN(in_channels=1, num_classes=10).to(device)
        network.apply(reset_weights)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        train_model(network, trainloader, optimizer, loss_function, num_epochs, device)

        print('Training process has finished. Saving trained model.')
        print('Starting testing')

        accuracy = test_model(network, testloader, device)
        results[fold] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_network = network

        print('--------------------------------')

    print(f'Best fold accuracy: {best_accuracy:.2f}% — saving C++ weights.')
    save_weights_for_cpp(best_network, './weights_cpp')

    return results

def save_weights_for_cpp(network, save_dir):
    """Dump each param/buffer as raw C-order float32 + a JSON manifest."""
    os.makedirs(save_dir, exist_ok=True)
    manifest = {}
    for name, tensor in network.state_dict().items():
        if name.endswith('num_batches_tracked'):   # int64 scalar, not needed
            continue
        arr = np.ascontiguousarray(tensor.detach().cpu().float().numpy())
        filename = name.replace('.', '_') + '.bin'
        arr.tofile(os.path.join(save_dir, filename))
        manifest[name] = {
            'shape': list(arr.shape),
            'count': int(arr.size),     # lets C++ validate file size == count*4
            'dtype': 'float32',
            'file': filename,
        }
    with open(os.path.join(save_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Saved {len(manifest)} tensors to {save_dir}/')


# Function to print k-fold cross-validation results
def print_results(results):
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {len(results)} FOLDS')
    print('--------------------------------')

    total_accuracy = sum(results.values())
    average_accuracy = total_accuracy / len(results)

    for key, value in results.items():
        print(f'Fold {key}: {value:.2f}%')

    print(f'Average: {average_accuracy:.2f}%')



# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--k_folds', type=int, default=K_FOLDS)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device)
    loss_function = nn.CrossEntropyLoss()

    results = k_fold_cross_validation(args.k_folds, args.num_epochs, loss_function, args.batch_size, device)
    print_results(results)


if __name__ == '__main__':
    main()
