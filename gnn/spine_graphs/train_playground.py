import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Embedding

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

from gnn.gdl_notebooks.train import run_experiment

from mid.data import Annotation
from mid.tests import read_test_data
from gnn.spine_graphs.endplate_graph import EndplateGraph
from gnn.spine_graphs.geometric_features import calc_spondy, calc_disc_height, get_endplate_geometric_data, check_if_lordotic
from gnn.spine_graphs.utils3d import calc_angle_between_vectors
from gnn.scripts.generate_dataset import generate_endplate_dataset

from scipy.stats import ortho_group

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))


class InvariantEndplateMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=2, geometric_feat_dim=10, aggr='add'):
        """Message Passing Neural Network Layer

        This layer is invariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + geometric_feat_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        # ==========================================

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

        pass

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """

        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)

        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """The `message()` function constructs messages from source nodes j
           to destination nodes i for each edge (i, j) in `edge_index`.

           Args:
               ...

           Returns:
               ...
        """

        # calculate geometric features
        # for each endplate:
        #  - distance, unit direction vector
        # for endplate pairs:
        #  - angle
        #  - height (should be from upper to lower endplate, how can we know ?)
        #  - spondy (should be of lower endplate, how can we know ?)

        start_i, end_i, distance_i, vector_i, unit_vector_i = get_endplate_geometric_data(pos_i)
        start_j, end_j, distance_j, vector_j, unit_vector_j = get_endplate_geometric_data(pos_j)

        angle = calc_angle_between_vectors(unit_vector_i, unit_vector_j, units='deg')
        is_lordotic = check_if_lordotic(pos_i, pos_j)

        spondy_signed, spondy_vector = calc_spondy(pos_i, pos_j)
        height, height_vector_lower_upper = calc_disc_height(pos_i, pos_j)

        geometric_features = torch.cat([distance_i, unit_vector_i, distance_j, unit_vector_j, angle, is_lordotic, spondy_signed, height], dim=1)

        msg_features = torch.cat([h_i, h_j, edge_attr, geometric_features], dim=1)
        msg = self.mlp_msg(msg_features)

        return msg

    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """

        aggr = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

        return aggr

    def update(self, aggr_out, h):
        """The `update()` function computes the final node features by combining the
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        upd = self.mlp_upd(upd_out)
        return upd

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class InvariantEndplateMPNNModel(Module):
    def __init__(self, num_layers=5, emb_dim=64, in_dim=14, edge_dim=2, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        # self.embedding = Embedding(100, emb_dim, padding_idx=0)

        # Stack of invariant MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(InvariantEndplateMPNNLayer(emb_dim, edge_dim, aggr='add'))

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

        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        # h = self.embedding(data.x) # (n, d_n) -> (n, d)

        for conv in self.convs:
            h = h + conv(h, data.pos, data.edge_index, data.edge_attr)  # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)


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
    if isinstance(module, InvariantEndplateMPNNModel):
        out_1 = module(data)
    else: # if isinstance(module, MessagePassing):
        out_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=2)
    t = torch.rand(2)
    # ============ YOUR CODE HERE ==============
    # Perform random rotation + translation on data.
    #
    # data.pos = ...

    data.pos[:, :2] = data.pos[:, :2] @ Q + t
    data.pos[:, 2:] = data.pos[:, 2:] @ Q + t
    # ==========================================

    # Forward pass on rotated + translated example
    if isinstance(module, InvariantEndplateMPNNModel):
        out_2 = module(data)
    else: # if isinstance(module, MessagePassing):
        out_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # ============ YOUR CODE HERE ==============
    # Check whether output varies after applying transformations.
    #
    # return torch.allclose(..., atol=1e-04)
    # ==========================================

    is_invariant = torch.allclose(out_1, out_2, atol=1e-04)

    return is_invariant



def simple_train():

    # ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]
    #
    # ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True)
    # keys_sorted = Annotation.sort_keys_by_vert_names(ann_dict.keys())
    # ann_dict = {key: ann_dict[key] for key in keys_sorted}
    #
    # graph = EndplateGraph(ann_dict, display=False)
    # dataset = [graph.pyg_graph]

    dataset = generate_endplate_dataset(n_max=5, s1_upper_only=False, projection='LT')

    # ==========================================
    # test model and layer invariance
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Rotation and translation invariance unit test for MPNN layer
    layer = InvariantEndplateMPNNLayer(emb_dim=1, edge_dim=2, geometric_feat_dim=10, aggr='add')
    print(f"Is {type(layer).__name__} rotation and translation invariant? --> {rot_trans_invariance_unit_test(layer, dataloader)}!")

    # Rotation and translation invariance unit test for MPNN model
    model = InvariantEndplateMPNNModel(num_layers=5, emb_dim=64, in_dim=1, edge_dim=2, out_dim=1)
    print(f"Is {type(model).__name__} rotation and translation invariant? --> {rot_trans_invariance_unit_test(model, dataloader)}!")
    # -------------------------------

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = InvariantEndplateMPNNModel(num_layers=5, emb_dim=64, in_dim=1, edge_dim=2, out_dim=1)

    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model"])
    model_name = type(model).__name__
    best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
        model,
        model_name,  # "MPNN w/ Features and Coordinates (Invariant Layers)",
        train_loader,
        val_loader,
        test_loader,
        n_epochs=1000
    )

    RESULTS[model_name] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Test MAE", "Val MAE", "Epoch", "Model"])
    # DF_RESULTS = DF_RESULTS.merge(df_temp, ignore_index=True)
    DF_RESULTS = pd.concat((DF_RESULTS, df_temp), ignore_index=True)

    print(RESULTS)

    sns.set_style('darkgrid')

    fig, ax = plt.subplots()
    p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 2))

    fig, ax = plt.subplots()
    p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))

    pass


if __name__ == '__main__':

    simple_train()

    pass