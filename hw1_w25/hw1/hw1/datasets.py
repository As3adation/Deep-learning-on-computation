import torch
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset


def random_labelled_image(
    shape: Tuple[int, ...], num_classes: int, low=0, high=255, dtype=torch.int,
) -> Tuple[Tensor, int]:
    """
    Generates a random image and a random class label for it.
    :param shape: The shape of the generated image e.g. (C, H, W).
    :param num_classes: Number of classes. The label should be in [0, num_classes-1].
    :param low: Minimal value in the image (inclusive).
    :param high: Maximal value in the image (exclusive).
    :param dtype: Data type of the returned image tensor.
    :return: A tuple containing the generated image tensor and it's label.
    """
    # TODO:
    #  Implement according to the docstring description.
    # ====== YOUR CODE: ======
    image = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    label = torch.randint(low=0, high=num_classes, size=(1,)).item()
    # ========================
    return image, label


@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """
    # TODO:
    #  Implement this context manager as described.
    #  See torch.random.get/set_rng_state(), torch.random.manual_seed().
    # ====== YOUR CODE: ======
    rng_state = torch.random.get_rng_state()
    # ========================
    try:
        # ====== YOUR CODE: ======
        torch.random.manual_seed(seed)
        # ========================
        yield
    finally:
        # ====== YOUR CODE: ======
        torch.random.set_rng_state(rng_state)
        # ========================


class RandomImageDataset(Dataset):
    """
    A dataset representing a set of noise images of specified dimensions.
    """

    def __init__(self, num_samples: int, num_classes: int, C: int, W: int, H: int):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Returns a labeled sample.
        :param index: Sample index.
        :return: A tuple (sample, label) containing the image and its class label.
        Raises a ValueError if index is out of range.
        """

        # TODO:
        #  Create a random image tensor and return it.
        #  Make sure to always return the same image for the
        #  same index (make it deterministic per index), but don't mess-up
        #  the random state outside this method.
        #  Raise a ValueError if the index is out of range.
        # ====== YOUR CODE: ======
        if (index < 0 or index >= self.num_samples):
            raise ValueError(f"Index {index} is out of range")
            
        with torch_temporary_seed(index):
            # Generate a random image tensor
            image = torch.rand(self.image_dim)
            # Assign a deterministic label for this index
            label = index % self.num_classes

        return image, label
        # ========================

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.num_samples
        # ========================


class ImageStreamDataset(IterableDataset):
    """
    A dataset representing an infinite stream of noise images of specified dimensions.
    """

    def __init__(self, num_classes: int, C: int, W: int, H: int):
        """
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_dim = (C, W, H)

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        """
        :return: An iterator providing an infinite stream of random labelled images.
        """

        # TODO:
        #  Yield tuples to produce an iterator over random images and labels.
        #  The iterator should produce an infinite stream of data.
        # ====== YOUR CODE: ======
        while True:
            # Generate a random seed based on the system time or a global counter
            seed = torch.randint(0, 2**31, (1,)).item()
            
            # Use torch_temporary_seed to ensure random generation without affecting global RNG
            with torch_temporary_seed(seed):
                # Generate a random image tensor
                image = torch.rand(self.image_dim)
                # Assign a random label
                label = seed % self.num_classes

            yield image, label
        # ========================


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """

    def __init__(self, source_dataset: Dataset, subset_len: int, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # TODO:
        #  Return the item at index + offset from the source dataset.
        #  Raise an IndexError if index is out of bounds.
        # ====== YOUR CODE: ======
        if(index < 0 or index >= self.subset_len):
            raise IndexError("Index out of range")
        return self.source_dataset[index + self.offset]
        # ========================

    def __len__(self):
        # ====== YOUR CODE: ======
        return self.subset_len
        # ========================
