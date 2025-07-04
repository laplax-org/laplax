import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

"""
    Module for providing easy access to common datasets + utility functions
"""


class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True, seed=0):  # noqa: FBT002
        self.X = jnp.array(X)
        self.y = jnp.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.X.shape[0]
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.indices = jnp.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            self.indices = jax.random.permutation(subkey, self.num_samples)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[
            self.current_idx : self.current_idx + self.batch_size
        ]
        batch_X = jnp.take(self.X, batch_indices, axis=0)
        batch_y = jnp.take(self.y, batch_indices, axis=0)
        self.current_idx += self.batch_size
        return batch_X, batch_y

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def fashion_mnist(
    batch_size=64, random_state=0, cache_dir=None
) -> tuple[DataLoader, DataLoader, int, int]:
    """Loads the Fashion-MNIST dataset from OpenML and caches it.

    Args:
        batch_size: Batch size of the resulting DataLoader
        random_state: seed for reproducibility
        cache_dir: cache location to save to/ load from

    Returns:
        trainloader, testloader, len(trainloder), len(testloader)
    """
    if cache_dir is None:
        cache_dir = Path.expanduser("~/.moml_cache/fashion_mnist")
    else:
        cache_dir = Path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    X_path = Path(cache_dir) / "X.npy"
    y_path = Path(cache_dir) / "y.npy"

    if X_path.exists() and y_path.exists():
        logging.info("[Fashion-MNIST] Loading from cache...")  # noqa: LOG015
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        logging.info("[Fashion-MNIST] Downloading from OpenML and caching...")  # noqa: LOG015
        data = fetch_openml(
            "Fashion-MNIST", version=1, as_frame=False, parser="liac-arff"
        )
        X, y = data["data"], data["target"].astype(np.int32)
        X = X.astype(np.float32) / 255.0
        np.save(X_path, X)
        np.save(y_path, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    y_train, y_test = jax.nn.one_hot(y_train, 10), jax.nn.one_hot(y_test, 10)

    trainloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, len(X_train), len(X_test)


def permute(dataloader: DataLoader, seed: int = 0) -> DataLoader:
    """Permutes a dataloader.

    Args:
        dataloader: DataLoader to be permuted.
        seed: random seed to be used to generate the permutation


    Returns:
        A new DataLoader where each image in the dataset is permuted
        using the same fixed random permutation (applied across all samples).
    """
    if seed == 0:
        return DataLoader(
            X=dataloader.X,
            y=dataloader.y,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
        )

    # Generate a fixed permutation of pixel indices
    num_pixels = dataloader.X.shape[1]
    key = jax.random.PRNGKey(seed)
    perm_indices = jax.random.permutation(key=key, x=num_pixels)
    X_permuted = dataloader.X[:, perm_indices]

    return DataLoader(
        X=X_permuted,
        y=dataloader.y,
        batch_size=dataloader.batch_size,
        shuffle=dataloader.shuffle,
        seed=seed,
    )


def collect(loader: DataLoader, maxsamples: int) -> tuple[Array, Array]:
    x, y = [], []
    cur = 0
    for xx, yy in loader:
        x.append(xx)
        y.append(yy)
        cur += len(xx)
        if cur >= maxsamples:
            break

    xret, yret = jnp.concat(x, axis=0), jnp.concat(y, axis=0)
    return xret[:maxsamples], yret[:maxsamples]


def minimnist(
    batch_size=64, random_state=0, cache_dir=None
) -> tuple[DataLoader, DataLoader, int, int]:
    """Loads a sub-sampled version of MNIST (6k train/test samples instead of 60k/10k).

    Args:
        batch_size: Batch size of the resulting DataLoader
        random_state: seed for reproducibility
        cache_dir: cache location to save to/ load from

    Returns:
        trainloader, testloader, len(trainloder), len(testloader)
    """
    if cache_dir is None:
        cache_dir = Path.expanduser("~/.moml_cache/minimnist")
    else:
        cache_dir = Path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    X_path = Path(cache_dir) / "X.npy"
    y_path = Path(cache_dir) / "y.npy"

    if X_path.exists() and y_path.exists():
        logging.info("[MiniMNIST] Loading from cache...")  # noqa: LOG015
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        logging.info("[MiniMNIST] Subsampling from MNIST and caching...")  # noqa: LOG015
        trainloader, testloader, *_ = mnist()
        X, y = (
            jnp.concat([trainloader.X, testloader.X]),
            jnp.concat([trainloader.y, testloader.y]),
        )
        y = jnp.argmax(y, axis=-1)
        _, X, _, y = train_test_split(
            X, y, test_size=0.1, random_state=random_state, stratify=y
        )
        y = jax.nn.one_hot(y, num_classes=10)
        np.save(X_path, X)
        np.save(y_path, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    trainloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, len(X_train), len(X_test)


def mnist(batch_size=64, random_state=0, cache_dir=None) \
    -> tuple[DataLoader, DataLoader]:
    """Loads MNIST using torchvision, but returns this file's DataLoader objects.

    Args:
        batch_size: Batch size of the resulting DataLoader
        random_state: seed
        cache_dir: cache location to save to/ load from

    Returns:
        trainloader and testloader
    """
    if cache_dir is None:
        cache_dir = Path.expanduser("~/.moml_cache/torch_mnist")
    else:
        cache_dir = Path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(
        root=cache_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=cache_dir, train=False, download=True, transform=transform
    )

    def dataset_to_arrays(dataset):
        data = [np.array(img).reshape(-1) for img, _ in dataset]
        targets = [target for _, target in dataset]
        data = np.stack(data)
        data = data.astype(np.float32)
        targets = np.array(targets)
        targets = jax.nn.one_hot(targets, num_classes=10)
        return data, targets

    X_train, y_train = dataset_to_arrays(train_dataset)
    X_test, y_test = dataset_to_arrays(test_dataset)

    trainloader = DataLoader(
        X_train, y_train, batch_size=batch_size, shuffle=True, seed=random_state
    )
    testloader = DataLoader(
        X_test, y_test, batch_size=batch_size, shuffle=False, seed=random_state
    )
    return trainloader, testloader, len(X_train), len(X_test)
