"""
Course page:
https://geometricdeeplearning.com/lectures/

Original colab notebook:
https://colab.research.google.com/drive/1Z0D10BFMdbsTM7lwPYrrJCe7z4yD48EE

My copy:
https://colab.research.google.com/drive/13m6BDOnouhyaw-CbjyBYwVDAjie8NqUx#scrollTo=EiuxXrwgmBE-&uniqifier=1
"""

import numpy as np
import seaborn as sns
import math
import itertools
import scipy as sp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid, ZINC, GNNBenchmarkDataset
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_geometric.utils import to_dense_adj
from torch.nn import Embedding
from torch.nn.modules.linear import Linear

import pdb

#for nice visualisations
import networkx as nx
import matplotlib.pyplot as plt

from mycolorpy import colorlist as mcp
import matplotlib.cm as cm

from typing import Mapping, Tuple, Sequence, List
import colorama

import scipy.linalg
from scipy.linalg import block_diag


from gnn.gdl_notebooks.utils import plot_stats, update_stats, gallery, print_color_numpy



class Graph(object):
    def __init__(self, edge_index, x, y):
        """ Graph structure
            for a mini-batch it will store a big (sparse) graph
            representing the entire batch
        Args:
            x: node features  [num_nodes x num_feats]
            y: graph labels   [num_graphs]
            edge_index: list of edges [2 x num_edges]
        """
        self.edge_index = edge_index
        self.x = x.to(torch.float32)
        self.y = y
        self.num_nodes = self.x.shape[0]

    #ignore this for now, it will be useful for batching
    def set_batch(self, batch):
        """ list of ints that maps each node to the graph it belongs to
            e.g. for batch = [0,0,0,1,1,1,1]: the first 3 nodes belong to graph_0 while
            the last 4 belong to graph_1
        """
        self.batch = batch

    # this function return a sparse tensor
    def get_adjacency_matrix(self):
        """ from the list of edges create
        a num_nodes x num_nodes sparse adjacency matrix
        """
        return torch.sparse.LongTensor(self.edge_index,
                              # we work with a binary adj containing 1 if an edge exist
                              torch.ones((self.edge_index.shape[1])),
                              torch.Size((self.num_nodes, self.num_nodes))
                              )


def create_mini_batch(graph_list: List[Graph]) -> Graph:
    """ Build a sparse graph from a batch of graphs
    Args:
        graph_list: list of Graph objects in a batch
    Returns:
        a big (sparse) Graph representing the entire batch
    """
    # insert first graph into the structure
    batch_edge_index = graph_list[0].edge_index
    batch_x = graph_list[0].x
    batch_y = graph_list[0].y
    batch_batch = torch.zeros((graph_list[0].num_nodes), dtype=torch.int64)
    # ============ YOUR CODE HERE =============
    # you may need additional variables
    # ==========================================
    # A = graph_list[0].get_adjacency_matrix()
    # A_list = [A]

    # append the rest of the graphs to the structure
    for idx, graph in enumerate(graph_list[1:]):
        # ============ YOUR CODE HERE =============
        # concat the features
        # batch_x = ...
        # concat the labels
        # batch_y = ...

        # concat the adjacency matrix as a block diagonal matrix
        # batch_edge_index = ...

        batch_x = torch.cat((batch_x, graph.x), dim=0)
        batch_y = torch.cat((batch_y, graph.y), dim=0)

        edge_index = graph.edge_index + len(batch_batch)
        batch_edge_index = torch.cat((batch_edge_index, edge_index), dim=1)
        # A = graph.get_adjacency_matrix()
        # A_list.append(A)

        # ==========================================

        # ============ YOUR CODE HERE =============
        # create the array of indexes mapping nodes in the batch-graph
        # to the graph they belong to
        # specify the mapping between the new nodes and the graph they belong to (idx+1)
        # batch_batch = ...
        # ==========================================

        batch = (idx + 1) * torch.ones((graph.num_nodes), dtype=torch.int64)
        batch_batch = torch.cat((batch_batch, batch), dim=0)

        pass

    # A_total = torch.block_diag(*A_list)
    # batch_edge_index = A_total

    # create the big sparse graph
    batch_graph = Graph(batch_edge_index, batch_x, batch_y)
    # attach the index array to the Graph structure
    batch_graph.set_batch(batch_batch)
    return batch_graph

class GINLayer(nn.Module):
    """A single GIN layer, implementing MLP(AX + (1+eps)X)"""
    def __init__(self, in_feats: int, out_feats: int, hidden_dim: int, eps: float=0.0):
        super(GINLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # ============ YOUR CODE HERE =============
        # epsilon should be a learnable parameter
        # self.eps = ...
        # =========================================
        self.eps = eps
        self.linear1 = nn.Linear(self.in_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.out_feats)

    def forward(self, x, adj_sparse):
        # ============ YOUR CODE HERE =============
        # aggregate the neighbours as in GIN: (AX + (1+eps)X)
        # x = ...
        x = adj_sparse @ x + (1 + self.eps) * x

        # project the features (MLP_k)
        # out =
        # =========================================
        x = self.linear1(x)
        out = self.linear2(x)

        return out

class SimpleGIN(nn.Module):
    """
    A Graph Neural Network containing GIN layers
    as in https://arxiv.org/abs/1810.00826
    The readout function used to obtain graph-lvl representations
    is just the sum of the nodes in the graph

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        num_layers (int): Number of layers
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, eps=0.0, molecular=True):
        super(SimpleGIN, self).__init__()
        self.num_layers = num_layers # please select num_layers>=2
        self.molecular = molecular
        # nodes in ZINC dataset are characterised by one integer (atom category)
        # we will create embeddings from the categorical features using nn.Embedding
        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        # instead of nn.Linear as in SimpleMLP model,
        # now we have (num_layers) GINLayer(s), each with different parameters
        self.layers = [GINLayer(hidden_dim, hidden_dim, hidden_dim, eps) for _ in range(num_layers-1)]
        self.layers += [GINLayer(hidden_dim, output_dim, hidden_dim, eps)]
        self.layers = nn.ModuleList(self.layers)

        pass

    def forward(self, graph):

        adj_sparse = graph.get_adjacency_matrix()
        if self.molecular:
            x = self.embed_x(graph.x.long()).squeeze(1)
        else:
            x = self.embed_x(graph.x)

        for i in range(self.num_layers-1):
            x = self.layers[i](x, adj_sparse)
            x = F.relu(x)
        x = self.layers[-1](x, adj_sparse)

        # ============ YOUR CODE HERE =============
        # graph-level representations are obtain by pooling info from the nodes using sum
        # y_hat = ...
        # =========================================
        ind = graph.batch
        y_hat = scatter_sum(x, ind, dim=0)

        y_hat = y_hat.squeeze(-1)
        #return also the final node embeddings (for visualisations)
        return y_hat, x

#@title [RUN] Unit test for mini-batch implementation
def unit_test_mini_batch(batch, BATCH_SIZE, HIDDEN_DIM):

    model = SimpleGIN(input_dim=batch[0].x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4)

    graph_batch = create_mini_batch(batch)
    out_batch, _ = model(graph_batch)

    for i in range(BATCH_SIZE):
        batch_i = create_mini_batch([batch[i]])
        out_i, node_emb_i = model(batch_i)
        assert(np.abs(out_i.detach().numpy() - out_batch[i].detach().numpy()).mean() < 1e-5 )
    print("Congrats 😊 !! Everything seems all right!")


def train(dataset, model, optimiser, epoch, loss_fct, metric_fct, print_every, BATCH_SIZE):
    """ Train model for one epoch
    """
    model.train()
    num_iter = int(len(dataset)/BATCH_SIZE)
    for i in range(num_iter):
        batch_list = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch = create_mini_batch(batch_list)
        optimiser.zero_grad()
        y_hat, _ = model(batch)
        loss = loss_fct(y_hat, batch.y)
        metric = metric_fct(y_hat, batch.y)
        loss.backward()
        optimiser.step()
        if (i+1) % print_every == 0:
          print(f"Epoch {epoch} Iter {i}/{num_iter}",
                f"Loss train {loss.data}; Metric train {metric.data}")
    return loss, metric

def evaluate(dataset, model, loss_fct, metrics_fct, BATCH_SIZE):
    """ Evaluate model on dataset
    """
    model.eval()
    # be careful in practice, as doing this way we will lose some
    # examples from the validation split, when len(dataset)%BATCH_SIZE != 0
    # think about how can you fix this!
    num_iter = int(len(dataset)/BATCH_SIZE)
    metrics_eval = 0
    loss_eval = 0
    num_samples = 0
    for i in range(num_iter):
        batch_list = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch = create_mini_batch(batch_list)
        y_hat, _ = model(batch)
        metrics = metrics_fct(y_hat, batch.y)
        loss = loss_fct(y_hat, batch.y)

        num_samples_in_batch = len(batch_list)
        num_samples += num_samples_in_batch
        metrics_eval += num_samples_in_batch * metrics.data
        loss_eval += num_samples_in_batch * loss.data

    metrics_eval /= num_samples
    loss_eval /= num_samples

    metrics_eval = metrics_eval.detach().numpy()
    loss_eval = loss_eval.detach().numpy()

    return loss_eval, metrics_eval

def train_eval(model, train_dataset, val_dataset, test_dataset,
               loss_fct, metric_fct, LR, BATCH_SIZE, NUM_EPOCHS, print_every=1):
    """ Train the model for NUM_EPOCHS epochs
    """
    #Instantiatie our optimiser
    optimiser = optim.Adam(model.parameters(), lr=LR)
    training_stats = None

    #initial evaluation (before training)
    val_loss, val_metric = evaluate(val_dataset, model, loss_fct, metric_fct, BATCH_SIZE)
    train_loss, train_metric = evaluate(train_dataset[:BATCH_SIZE], model, loss_fct, metric_fct, BATCH_SIZE)
    epoch_stats = {'train_loss': train_loss, 'val_loss': val_loss, 'train_metric': train_metric, 'val_metric': val_metric, 'epoch':0}
    training_stats = update_stats(training_stats, epoch_stats)

    for epoch in range(NUM_EPOCHS):
        if isinstance(train_dataset, list):
            random.shuffle(train_dataset)
        else:
            train_dataset.shuffle()
        train_loss, train_metric = train(train_dataset, model, optimiser, epoch, loss_fct, metric_fct, print_every, BATCH_SIZE)
        val_loss, val_metric = evaluate(val_dataset, model, loss_fct, metric_fct, BATCH_SIZE)
        print(f"[Epoch {epoch+1}]",
              f"train loss: {train_loss:.3f} val loss: {val_loss:.3f}",
              f"train metric: {train_metric:.3f} val metric: {val_metric:.3f}"
              )
        # store the loss and the computed metric for the final plot
        epoch_stats = {'train_loss': train_loss.detach().numpy(), 'val_loss': val_loss, 'train_metric': train_metric.detach().numpy(), 'val_metric': val_metric, 'epoch':epoch+1}
        training_stats = update_stats(training_stats, epoch_stats)

    test_loss, test_metric = evaluate(test_dataset, model,  loss_fct, metric_fct, BATCH_SIZE)
    print(f"Test metric: {test_metric:.3f}")
    return training_stats

class GIN(nn.Module):
    """
    A Graph Neural Network containing GIN layers
    as in https://arxiv.org/abs/1810.00826
    The readout function used to obtain graph-lvl representations
    aggregate pred from multiple layers (as in JK-Net)

    Args:
    input_dim (int): Dimensionality of the input feature vectors
    output_dim (int): Dimensionality of the output softmax distribution
    num_layers (int): Number of layers
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, eps=0.0, molecular=True):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.molecular = molecular
        # nodes in ZINC dataset are characterised by one integer (atom category)
        # we will create embeddings from the categorical features using nn.Embedding
        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = Linear(input_dim, hidden_dim)

        # ============ YOUR CODE HERE =============
        # should be the same as before (an nn.ModuleList of GINLayers)
        # self.layers = ...
        self.layers = [GINLayer(hidden_dim, hidden_dim, hidden_dim, eps) for _ in range(num_layers-1)]
        self.layers += [GINLayer(hidden_dim, output_dim, hidden_dim, eps)]
        self.layers = nn.ModuleList(self.layers)


        # layer to compute prediction from the concatenated intermediate representations
        # self.pred_layers = ...
        # =========================================
        self.pred_layers = [nn.Linear(input_dim, output_dim)]
        # self.pred_layers += [nn.Linear(hidden_dim, output_dim) for _ in range(num_layers - 1)]
        self.pred_layers += [nn.Linear(hidden_dim, output_dim) for _ in range(num_layers)]
        # self.pred_layers = [nn.Linear(hidden_dim, output_dim) for _ in range(num_layers)]
        self.pred_layers = nn.ModuleList(self.pred_layers)
        # self.pred_layers = nn.Linear((num_layers-1) * hidden_dim, output_dim)
        self.drop = nn.Dropout(0.4)

        pass

    def forward(self, graph):
        adj_sparse = graph.get_adjacency_matrix()
        if self.molecular:
            x = self.embed_x(graph.x.long()).squeeze(1)
        else:
            x = self.embed_x(graph.x)

        # ============ YOUR CODE HERE =============
        # perform the forward pass with the new readout function
        hidden_rep = [graph.x.long(), x]
        for i in range(self.num_layers):
            # x = ...
            x = self.layers[i](x, adj_sparse)
            x = F.relu(x)
            hidden_rep.append(x)
            pass

        # loop over all x beside the last on
        ind = graph.batch
        score_over_layer = 0
        for i in range(0, (self.num_layers)):
            h = hidden_rep[i]
            pooled_h = scatter_sum(h, ind, dim=0)
            hg = self.pred_layers[i](pooled_h)
            hg = self.drop(hg)
            score_over_layer += hg
            pass

        # hg_concat = torch.cat(hg_list, dim=0)

        # y_hat = ...
        # ind = graph.batch
        # y_hat = scatter_sum(hg_concat, ind, dim=0)
        # y_hat = hg_concat.sum(dim=0)
        y_hat = score_over_layer.squeeze()

        # ind = graph.batch
        # h_list = []
        # for i in range(self.num_layers-1):
        #     # x = ...
        #     x = self.layers[i](x, adj_sparse)
        #     x = F.relu(x)
        #     h = scatter_sum(x, ind, dim=0)
        #     h_list.append(h)
        #     pass
        #
        # hg = torch.cat(h_list, dim=1)
        # y_hat = self.pred_layers(hg)

        pass
        # =========================================
        # return also the final node embeddings (for visualisations)
        return y_hat, x


def main():

    train_zinc_dataset = ZINC(root='', split='train', subset=True)
    val_zinc_dataset = ZINC(root='', split='val', subset=True)
    test_zinc_dataset = ZINC(root='', split='test', subset=True)

    print(f"\nTrain examples: {len(train_zinc_dataset)}")
    print(f"Val examples: {len(val_zinc_dataset)}")
    print(f"Test examples: {len(test_zinc_dataset)}\n")

    one_graph = train_zinc_dataset[0]

    print(f"First graph contains {one_graph.x.shape[0]} nodes, each characterised by {one_graph.x.shape[1]} features")
    print(f"Graph labels have shape: {one_graph.y.shape}")

    # visualize graph
    gallery([one_graph], labels=np.array([one_graph.y]), max_fig_size=(8, 10))

    # ------------------------
    # Mini-batching graphs
    # ------------------------

    print(f'First graph : {train_zinc_dataset[0].x.shape} with adjacency {(train_zinc_dataset[0].num_nodes, train_zinc_dataset[0].num_nodes)}')
    print(f'Second graph: {train_zinc_dataset[1].x.shape} with adjacency {(train_zinc_dataset[1].num_nodes, train_zinc_dataset[1].num_nodes)}')
    print(f'Third graph : {train_zinc_dataset[2].x.shape} with adjacency {(train_zinc_dataset[2].num_nodes, train_zinc_dataset[2].num_nodes)}')

    # @title Visualize the mini-batching for a small list of batch_size=3 graphs.
    # Note that the three graphs viusalized are directed,
    # so the adjacency matrix will be non-symmetric
    # (even if the visualisation depicted them as undirected)

    # 3 random custom-designed graphs for visualisations
    graph1 = Graph(x=torch.rand((3, 32)),
                   y=torch.rand((1)),
                   edge_index=torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
    graph2 = Graph(x=torch.rand((5, 32)),
                   y=torch.rand((1)),
                   edge_index=torch.tensor(
                       [[0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 2, 3, 4], [0, 1, 2, 3, 4, 2, 3, 4, 4, 0, 0, 0, 0]]))
    graph3 = Graph(x=torch.rand((4, 32)),
                   y=torch.rand((1)),
                   edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
    list_graphs = [graph1, graph2, graph3]

    # create a mini-batch from these 3 graphs
    batch_sample = create_mini_batch(list_graphs)

    # show statistics about the new graph built from this batch of graphs
    print(f"Batch number_of_nodes: {batch_sample.num_nodes}")
    print(f"Batch features shape: {batch_sample.x.shape}")
    print(f"Batch labels shape: {batch_sample.y.shape}")

    print(f"Graph1 adjacency: ")
    print(graph1.get_adjacency_matrix().to_dense().numpy())
    print()
    print(f"Graph2 adjacency: ")
    print(graph2.get_adjacency_matrix().to_dense().numpy())
    print()
    print(f"Graph3 adjacency: ")
    print(graph3.get_adjacency_matrix().to_dense().numpy())
    print()

    print(f"Batch adjacency: ")
    print_color_numpy(batch_sample.get_adjacency_matrix().to_dense().numpy(), list_graphs)

    gallery([graph1, graph2, graph3, batch_sample], max_fig_size=(20, 6), special_color=True)
    print(f"And we also have access to which graph each node belongs to {batch_sample.batch}\n")

    # ----------------------------------
    # Scatter for aggregate information
    # ----------------------------------
    array = torch.tensor([13, 21, 3, 7, 11, 20, 2])
    index = torch.tensor([0, 1, 1, 0, 2, 0, 1])

    aggregate_sum = scatter_sum(array, index, dim=0)
    aggregate_mean = scatter_mean(array, index, dim=0)
    aggregate_max, aggregate_argmax = scatter_max(array, index, dim=0)

    print("Let's inspect what different scatter functions compute: ")
    print(f"sum aggregation: {aggregate_sum}")
    print(f"mean aggregation: {aggregate_mean}")
    print(f"max aggregation: {aggregate_max}\n")

    batch_zinc = create_mini_batch(train_zinc_dataset[:3])
    # ============ YOUR CODE HERE =============
    # Given the nodes features for a batch of graphs (batch_zinc.x)
    # and the list of indices indicating what graph each node belongs to
    # apply scatter_* to obtain a graph embedings for each graph in the batch
    # You can play with all of them (scatter_mean/scatter_max/scatter_sum)

    # node_emb = ...
    # node_batch = ...
    # graph_emb = ...
    # ==========================================

    node_emb = batch_zinc.x
    node_batch = batch_zinc.batch
    graph_emb = scatter_sum(node_emb, node_batch, dim=0)

    print(node_emb.shape)
    print(graph_emb.shape)

    # -----------------------------------------------
    # Graph Neural Network for graph-level regression
    # -----------------------------------------------

    BATCH_SIZE = 128  # @param {type:"integer"}
    NUM_EPOCHS = 30  # @param {type:"integer"}
    HIDDEN_DIM = 64  # @param {type:"integer"}
    LR = 0.001  # @param {type:"number"}
    # you can add more here if you need

    # @title Run unit test for mini-batch implementation
    batch = train_zinc_dataset[:BATCH_SIZE]
    unit_test_mini_batch(batch, BATCH_SIZE, HIDDEN_DIM)

    # Instantiate our SimpleGIN model
    model_simple_gin = SimpleGIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)
    out, _ = model_simple_gin(batch_zinc)
    print(out.detach().numpy())

    # # Train SimpleGIN model:
    # train_stats_simple_gin_zinc = train_eval(model_simple_gin, train_zinc_dataset, val_zinc_dataset,
    #                                          test_zinc_dataset, loss_fct=F.mse_loss,
    #                                          metric_fct=F.mse_loss,
    #                                          LR=LR, BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS,
    #                                          print_every=150)
    # plot_stats(train_stats_simple_gin_zinc, name='Simple_GIN_ZINC', figsize=(5, 10))

    # --------------------------------
    # now let's try a full GIN model
    # --------------------------------
    BATCH_SIZE = 128  # @param {type:"integer"}
    NUM_EPOCHS = 30  # @param {type:"integer"}
    HIDDEN_DIM = 64  # @param {type:"integer"}
    LR = 0.001  # @param {type:"number"}

    model_gin = GIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)
    out, _ = model_gin(batch_zinc)
    print(out.detach().numpy())

    # Train GIN model:
    train_stats_gin_zinc = train_eval(model_gin, train_zinc_dataset, val_zinc_dataset,
                                      test_zinc_dataset, loss_fct=F.mse_loss,
                                      metric_fct=F.mse_loss,
                                      LR=LR, BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS,
                                      print_every=150)
    plot_stats(train_stats_gin_zinc, name='GIN_ZINC', figsize=(5, 10))

    pass


if __name__ == '__main__':

    main()

    pass