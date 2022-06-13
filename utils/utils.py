import numpy as np
import os
import torch

def get_base_features(num_features, feature_dim, loop2edge = None, edge_type = None, type = "random"):
    if type == "random" or loop2edge == None or edge_type == None:
        X = torch.rand(num_features, feature_dim)
    elif type == "type_cnt":
        X = torch.matmul(loop2edge, edge_type)

    return X

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]