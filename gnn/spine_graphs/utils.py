import numpy as np


def get_one_hot_dict(id2data_dict):
    """
    Get one-hot row embedding for each item in id2data_dict.
    """

    ids = list(id2data_dict.keys())
    N = len(ids)
    one_hot_matrix = np.identity(N)
    one_hot_dict = {id: one_hot_matrix[n, :] for n, id in enumerate(ids)}

    return one_hot_dict