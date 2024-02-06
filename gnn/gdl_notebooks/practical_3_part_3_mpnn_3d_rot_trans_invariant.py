"""
Course page:
https://geometricdeeplearning.com/lectures/

Original colab notebook:
https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR

My copy:
https://colab.research.google.com/drive/1UW-rfX-IKa4TCXF-vhjSNJmc-E-nZ-pw#scrollTo=2xcV8Yb148Kq
"""

#@title [RUN] Import python modules

import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_scatter import scatter

import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import QED, Crippen, rdMolDescriptors, rdmolops

import py3Dmol
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))


from gnn.gdl_notebooks.practical_3_part_0b_pyg_message_passing import MPNNLayer, MPNNModel, permutation_invariance_unit_test, permutation_equivariance_unit_test
from gnn.gdl_notebooks.utils import get_qm9_data
from gnn.gdl_notebooks.train import run_experiment


def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q


def rot_trans_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model/layer) is
    rotation and translation invariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Forward pass on original example
    # Note: We have written a conditional forward pass so that the same unit
    #       test can be used for both the GNN model as well as the layer.
    #       The functionality for layers will be useful subsequently.
    if isinstance(module, MPNNModel):
        out_1 = module(data)
    else: # if ininstance(module, MessagePassing):
        out_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    # ============ YOUR CODE HERE ==============
    # Perform random rotation + translation on data.
    #
    # data.pos = ...
    # ==========================================

    # Forward pass on rotated + translated example
    if isinstance(module, MPNNModel):
        out_2 = module(data)
    else: # if ininstance(module, MessagePassing):
        out_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # ============ YOUR CODE HERE ==============
    # Check whether output varies after applying transformations.
    #
    # return torch.allclose(..., atol=1e-04)
    # ==========================================


class CoordMPNNModel(MPNNModel):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, coord_dim=3):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
            coord_dim: (int) - coordination dimension (3 - for 3D coordinates)
        """
        super().__init__()

        # ============ YOUR CODE HERE ==============
        # Adapt the input linear layer or add new input layers
        # to account for the atom positions.
        #
        # Linear projection for initial node features and coordinates
        # dim: ??? -> d
        # self.lin_in = ...

        self.lin_in = Linear(in_dim + coord_dim, emb_dim)

        # ==========================================

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        # ============ YOUR CODE HERE ==============
        # Incorporate the atom positions along with the features.
        #
        # h = ...

        node_feat = torch.cat((data.x, data.pos), dim=1)
        h = self.lin_in(node_feat)

        # ==========================================

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)


def main():

    train_dataset, val_dataset, test_dataset, std = get_qm9_data()

    # Instantiate temporary model, layer, and dataloader for unit testing
    model = CoordMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Rotation and translation invariance unit test for MPNN model
    print(f"Is {type(model).__name__} rotation and translation invariant? --> {rot_trans_invariance_unit_test(model, dataloader)}!")

    pass


if __name__ == '__main__':

    main()

    pass
