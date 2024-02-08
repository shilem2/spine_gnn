from scipy.stats import ortho_group
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.nn.pool import global_mean_pool

from torch_scatter import scatter

from gnn.gdl_notebooks.practical_3_part_0b_pyg_message_passing import MPNNLayer, MPNNModel
from gnn.gdl_notebooks.practical_3_part_3_mpnn_3d_rot_trans_invariant import rot_trans_invariance_unit_test
from gnn.gdl_notebooks.utils import get_qm9_data


class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, coord_dim=3, aggr='add'):
        """Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.coord_dim = coord_dim

        # ============ YOUR CODE HERE ==============
        # Define the MLPs constituting your new layer.
        # At the least, you will need `\psi` and `\phi`
        # (but their definitions may be different from what
        # we used previously).
        #
        # self.mlp_msg = ...  # MLP `\psi`
        # self.mlp_upd = ...  # MLP `\phi`
        # ===========================================

        # input: h_i, h_j node embeddings, edge embeddings
        self.mlp_msg_invariant = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

        # input: pos_j - pos_i (3d vector)
        self.mlp_msg_equivariant = Sequential(
            Linear(coord_dim, coord_dim), BatchNorm1d(coord_dim), ReLU(),
            Linear(coord_dim, coord_dim), BatchNorm1d(coord_dim), ReLU()
          )

        # input: h_i (current embedding), aggregated invariant message
        self.mlp_upd_invariant = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

        # input: pos_i (3d vector), aggregated equivariant message (3d vector)
        # the current position pos_i should introduce the translation equivariance.
        # the message composed of vector pointing from node i to node j should introduce the rotation equivariace.
        self.mlp_upd_equivariant = Sequential(
            Linear(2*coord_dim, coord_dim), BatchNorm1d(coord_dim), ReLU(),
            Linear(coord_dim, coord_dim), BatchNorm1d(coord_dim), ReLU()
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
            out: [(n, d),(n,3)] - updated node features
        """
        # ============ YOUR CODE HERE ==============
        # Notice that the `forward()` function has a new argument
        # `pos` denoting the initial node coordinates. Your task is
        # to update the `propagate()` function in order to pass `pos`
        # to the `message()` function along with the other arguments.
        #
        # out = self.propagate(...)
        # return out
        # ==========================================

        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    # ============ YOUR CODE HERE ==============
    # Write custom `message()`, `aggregate()`, and `update()` functions
    # which ensure that the layer is 3D rotation and translation equivariant.
    #
    # def message(self, ...):
    #   ...
    #
    # def aggregate(self, ...):
    #   ...
    #
    # def update(self, ...):
    #   ...
    #
    # ==========================================

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):

        msg_pos_input = pos_j - pos_i  # vector from i to j, equivariant to rotation
        msg_pos = self.mlp_msg_equivariant(msg_pos_input)

        msg_h_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        msg_h = self.mlp_msg_invariant(msg_h_input)

        return msg_h, msg_pos

    def aggregate(self, inputs, index):

        msg_h, msg_pos = inputs

        aggregated_msg_h = scatter(msg_h, index, dim=self.node_dim, reduce=self.aggr)
        aggregated_msg_pos = scatter(msg_pos, index, dim=self.node_dim, reduce=self.aggr)

        return aggregated_msg_h, aggregated_msg_pos

    def update(self, aggr_out, h, pos):
        """The `update()` function computes the final node features by combining the
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """

        aggregated_msg_h, aggregated_msg_pos = aggr_out

        update_input_h = torch.cat([h, aggregated_msg_h], dim=-1)
        update_h = self.mlp_upd_invariant(update_input_h)

        update_input_pos = torch.cat([pos, aggregated_msg_pos], dim=-1)
        update_pos = self.mlp_upd_equivariant(update_input_pos)

        return update_h, update_pos


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class FinalMPNNModel(MPNNModel):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, coord_dim=3, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            coord_dim: (int) - coordinate feature dimension
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        for layer in range(num_layers):
            self.convs = torch.nn.ModuleList()
            self.convs.append(EquivariantMPNNLayer(emb_dim, edge_dim, coord_dim, aggr='add'))

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
        pos = data.pos

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)

            # Update node features
            h = h + h_update  # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

            # Update node coordinates
            pos = pos_update  # (n, 3) -> (n, 3)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)


def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q

def rot_trans_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is
    rotation and translation equivariant.
    """
    it = iter(dataloader)
    data = next(it)

    out_1, pos_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    # ============ YOUR CODE HERE ==============
    # Perform random rotation + translation on data.
    #
    # data.pos = ...

    data.pos = data.pos @ Q #+ t

    # ==========================================

    # Forward pass on rotated + translated example
    out_2, pos_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # ============ YOUR CODE HERE ==============
    # Check whether output varies after applying transformations.
    # return ...
    # ==========================================

    is_invariant = torch.allclose(out_1, out_2, atol=1e-04)
    is_equivariant = torch.allclose(pos_1 @ Q, pos_2, atol=1e-04)

    return is_invariant and is_equivariant

def main():

    train_dataset, val_dataset, test_dataset, std = get_qm9_data()

    # Unit test FinalMPNNModel
    # Instantiate temporary model, layer, and dataloader for unit testing
    layer = EquivariantMPNNLayer(emb_dim=11, edge_dim=4, coord_dim=3, aggr='add')
    model = FinalMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, coord_dim=3, out_dim=1)

    # ============ YOUR CODE HERE ==============
    # Instantiate temporary model, layer, and dataloader for unit testing.
    # Remember that we are now unit testing the FinalMPNNModel,
    # which is  composed of the EquivariantMPNNLayer.
    #
    # layer = ...
    # model = ...
    # ==========================================
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Rotation and translation invariance unit test for MPNN model
    print(f"Is {type(model).__name__} rotation and translation invariant? --> {rot_trans_invariance_unit_test(model, dataloader)}!")

    # Rotation and translation invariance unit test for MPNN layer
    print(f"Is {type(layer).__name__} rotation and translation equivariant? --> {rot_trans_equivariance_unit_test(layer, dataloader)}!")

    pass


if __name__ == '__main__':

    main()

    pass