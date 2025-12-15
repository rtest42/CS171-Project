import numpy as np

from train_test import load_labels


def get_invperm(filename: str = "pems+sf/randperm"):
    randperm = load_labels(filename)

    # Compute inverse permutation
    inv_perm = np.empty_like(randperm)
    inv_perm[randperm] = np.arange(len(randperm))

    return inv_perm
