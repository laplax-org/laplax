# Compare dataloading from mnist_permute (PyTorch DataLoader) and runexp (vanilla numpy/jax)
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from richard_workspace.helpers import load_mnist as load_mnist_vanilla, permute_data, permute_loaders, load_mnist

def get_avg_digits_pytorch(train_loader, seed):
    permuted_train, _ = permute_loaders(train_loader, train_loader, seed)
    # Accumulate sums and counts for each digit
    sums = np.zeros((10, 28*28))
    counts = np.zeros(10)
    for data, target in permuted_train:
        # target: one-hot, shape (batch, 10)
        labels = np.argmax(np.array(target), axis=1)
        data = np.array(data)
        for i in range(10):
            mask = labels == i
            if np.any(mask):
                sums[i] += data[mask].sum(axis=0)
                counts[i] += mask.sum()
    avgs = sums / np.maximum(counts[:, None], 1)
    return avgs

def get_avg_digits_vanilla(xtrain, ytrain):
    # ytrain: one-hot, shape (N, 10)
    labels = np.argmax(np.array(ytrain), axis=1)
    data = np.array(xtrain)
    sums = np.zeros((10, 28*28))
    counts = np.zeros(10)
    for i in range(10):
        mask = labels == i
        if np.any(mask):
            sums[i] += data[mask].sum(axis=0)
            counts[i] += mask.sum()
    avgs = sums / np.maximum(counts[:, None], 1)
    return avgs

def main():
    # PyTorch DataLoader
    train_loader, _ = load_mnist(batch_size=128, num_workers=0)
    # Vanilla (jax/numpy)
    xtrain, ytrain, _, _ = load_mnist_vanilla()
    
    seeds = range(5)
    avg_digits_pytorch = []
    avg_digits_vanilla = []
    for seed in seeds:
        # PyTorch
        avg_digits_pytorch.append(get_avg_digits_pytorch(train_loader, seed))
        # Vanilla
        xperm, yperm = permute_data(xtrain, ytrain, seed=seed)
        avg_digits_vanilla.append(get_avg_digits_vanilla(xperm, yperm))
    avg_digits_pytorch = np.stack(avg_digits_pytorch)  # (5, 10, 784)
    avg_digits_vanilla = np.stack(avg_digits_vanilla)  # (5, 10, 784)

    # Plot 1: PyTorch loader, 5x10 grid (seeds x digits)
    fig1, axes1 = plt.subplots(5, 10, figsize=(20, 10))
    for i in range(5):
        for digit in range(10):
            axes1[i, digit].imshow(avg_digits_pytorch[i, digit].reshape(28, 28), cmap='gray', vmin=avg_digits_pytorch.min(), vmax=avg_digits_pytorch.max())
            axes1[i, digit].set_xticks([])
            axes1[i, digit].set_yticks([])
            if i == 0:
                axes1[i, digit].set_title(f'Digit {digit}')
            if digit == 0:
                axes1[i, digit].set_ylabel(f'Seed {i}')
    fig1.suptitle('PyTorch Loader: Mean Permuted Digits (Seeds 0-4)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('compare_dataloading_pytorch.png')

    # Plot 2: Vanilla loader, 5x10 grid (seeds x digits)
    fig2, axes2 = plt.subplots(5, 10, figsize=(20, 10))
    for i in range(5):
        for digit in range(10):
            axes2[i, digit].imshow(avg_digits_vanilla[i, digit].reshape(28, 28), cmap='gray', vmin=avg_digits_vanilla.min(), vmax=avg_digits_vanilla.max())
            axes2[i, digit].set_xticks([])
            axes2[i, digit].set_yticks([])
            if i == 0:
                axes2[i, digit].set_title(f'Digit {digit}')
            if digit == 0:
                axes2[i, digit].set_ylabel(f'Seed {i}')
    fig2.suptitle('Vanilla Loader: Mean Permuted Digits (Seeds 0-4)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('compare_dataloading_vanilla.png')

    plt.show()

if __name__ == '__main__':
    main()
