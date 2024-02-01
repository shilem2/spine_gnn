"""
Course page:
https://geometricdeeplearning.com/lectures/

Original colab notebook:
https://colab.research.google.com/drive/1Z0D10BFMdbsTM7lwPYrrJCe7z4yD48EE

My copy:
https://colab.research.google.com/drive/13m6BDOnouhyaw-CbjyBYwVDAjie8NqUx#scrollTo=EiuxXrwgmBE-&uniqifier=1
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.gdl_notebooks.utils import plot_stats, update_stats


# Let's get the Planetoid Cora dataset from
# “FastGCN: Fast Learning with Graph Convolutional
# Networks via Importance Sampling” (https://arxiv.org/abs/1801.10247)

class CoraDataset(object):
    def __init__(self):
        super(CoraDataset, self).__init__()
        cora_pyg = Planetoid(root='/tmp/Cora', name='Cora', split="full")
        self.cora_data = cora_pyg[0]
        self.train_mask = self.cora_data.train_mask
        self.valid_mask = self.cora_data.val_mask
        self.test_mask = self.cora_data.test_mask

    def train_val_test_split(self):
        train_x = self.cora_data.x[self.cora_data.train_mask]
        train_y = self.cora_data.y[self.cora_data.train_mask]

        valid_x = self.cora_data.x[self.cora_data.val_mask]
        valid_y = self.cora_data.y[self.cora_data.val_mask]

        test_x = self.cora_data.x[self.cora_data.test_mask]
        test_y = self.cora_data.y[self.cora_data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.cora_data.x

    def get_adjacency_matrix(self):
        # We will ignore this for the first part
        adj = to_dense_adj(self.cora_data.edge_index)[0]
        return adj


# Lets implement a simple feed forward MLP
class SimpleMLP(nn.Module):
    """A simple feed forward neural network with no hidden layers

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
    """

    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        y_hat = x
        return y_hat


# Lets define some utility functions for training and computing performance metrics
# and then see how our model does!
def train_mlp_cora(x, y, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data


def evaluate_mlp_cora(x, y, model):
    model.eval()
    y_hat = model(x)
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct / num_total)
    return accuracy


def train_eval_loop(model, train_x, train_y, valid_x, valid_y, test_x, test_y, NUM_EPOCHS, LR):
    optimiser = optim.Adam(model.parameters(), lr=LR)
    training_stats = None
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_mlp_cora(train_x, train_y, model, optimiser)
        train_acc = evaluate_mlp_cora(train_x, train_y, model)
        valid_acc = evaluate_mlp_cora(valid_x, valid_y, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f}",
                  f"validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch': epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc = evaluate_mlp_cora(test_x, test_y, model)
    print(f"Our final test accuracy for the SimpleMLP is: {test_acc:.3f}")
    return training_stats

# Fill in initialisation and forward method the GCNLayer below
class GCNLayer(nn.Module):
    """GCN layer to be implemented by students of practical

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # ============ YOUR CODE HERE =============
        # Compute symmetric norm
        # self.adj_norm = ...
        A_norm = self.A + torch.eye(A.shape[0])
        D_diag = A_norm.sum(axis=1)
        # option 1
        # D = torch.diag(D_diag)
        # D_pow_half = torch.pow(D, 0.5)
        # D_pow_half_inverse = torch.inverse(D_pow_half)
        # option 2
        D_pow_half_inverse = torch.diag(1 / torch.sqrt(D_diag))

        self.adj_norm = torch.Tensor(D_pow_half_inverse @ A_norm @ D_pow_half_inverse)

        # + Simple linear transformation and non-linear activation
        # self.linear = ...
        self.linear = nn.Linear(input_dim, output_dim)
        # =========================================

    def forward(self, x):
        # ============ YOUR CODE HERE =============
        # x = ...
        # =========================================

        x = self.linear(x)
        x = self.adj_norm @ x
        x = nn.functional.relu(x)

        return x

# Lets see the GCNLayer in action!
class SimpleGNN(nn.Module):
    """Simple GNN model using the GCNLayer implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.gcn_layer = GCNLayer(input_dim, output_dim, A)

    def forward(self, x):
        x = self.gcn_layer(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        y_hat = x
        return y_hat

def train_gnn_cora(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data

def evaluate_gnn_cora(X, y, mask, model):
    model.eval()
    y_hat = model(X)[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct/num_total)
    return accuracy

# Training loop
def train_eval_loop_gnn_cora(model, train_x, train_y, train_mask,
                             valid_x, valid_y, valid_mask,
                             test_x, test_y, test_mask,
                             LR, NUM_EPOCHS,
                    ):
    optimiser = optim.Adam(model.parameters(), lr=LR)
    training_stats = None
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_gnn_cora(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gnn_cora(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gnn_cora(valid_x, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc = evaluate_gnn_cora(test_x, test_y, test_mask, model)
    print(f"Our final test accuracy for the SimpleGNN is: {test_acc:.3f}")
    return training_stats

class SimpleGNN_2layers(nn.Module):
    """Simple GNN model using the GCNLayer implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(SimpleGNN_2layers, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.gcn_layer1 = GCNLayer(input_dim, 16, A)
        self.dropout = nn.Dropout(p=0.5)
        self.gcn_layer2 = GCNLayer(16, output_dim, A)

    def forward(self, x):
        x = self.gcn_layer1(x)
        x = self.dropout(x)
        # x = self.gcn_layer2(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        y_hat = x
        return y_hat



def main():
    # Lets download our cora dataset and get the splits
    cora_data = CoraDataset()
    train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

    # Always check and confirm our data shapes match our expectations
    print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
    print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
    print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")


    # ------------------
    # --- Simple MLP ---
    # ------------------
    print('\nSimple MLP\n')
    NUM_EPOCHS = 100  # @param {type:"integer"}
    LR = 0.001

    # Instantiate our model
    model = SimpleMLP(input_dim=train_x.shape[-1], output_dim=7)

    # Run training loop
    train_stats_mlp_cora = train_eval_loop(model, train_x, train_y, valid_x, valid_y, test_x, test_y, NUM_EPOCHS, LR)
    plot_stats(train_stats_mlp_cora, name="MLP_Cora")

    # ------------------
    # --- Simple GNN ---
    # ------------------

    print('\nSimple GNN\n')
    NUM_EPOCHS = 150
    LR = 0.001

    # Instantiate our model and optimiser
    A = cora_data.get_adjacency_matrix()
    X = cora_data.get_fullx()
    model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=A)

    train_mask = cora_data.train_mask
    valid_mask = cora_data.valid_mask
    test_mask = cora_data.test_mask

    # Run training loop
    train_stats_gnn_cora = train_eval_loop_gnn_cora(
        model, X, train_y, train_mask,
        X, valid_y, valid_mask,
        X, test_y, test_mask,
        LR, NUM_EPOCHS
    )
    plot_stats(train_stats_gnn_cora, name="GNN_Cora")

    # try model with 2 GCN layers
    print('\nSimple GNN - 2 layers\n')
    model = SimpleGNN_2layers(input_dim=train_x.shape[-1], output_dim=7, A=A)

    train_stats_gnn_cora = train_eval_loop_gnn_cora(
        model, X, train_y, train_mask,
        X, valid_y, valid_mask,
        X, test_y, test_mask,
        LR, NUM_EPOCHS
    )
    plot_stats(train_stats_gnn_cora, name="GNN_Cora - 2 Layers")


    pass


if __name__ == '__main__':
    main()

    pass
