"""
Course page:
https://geometricdeeplearning.com/lectures/

Original colab notebook:
https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR

My copy:
https://colab.research.google.com/drive/1UW-rfX-IKa4TCXF-vhjSNJmc-E-nZ-pw#scrollTo=ExJ0b3xcQl5n
"""

import torch
import itertools
import numpy as np

from gnn.gdl_notebooks.practical_1_part_2_graph_level_preds import Graph, create_mini_batch, SimpleGIN, ZINC
from gnn.gdl_notebooks.utils import gallery


#@title [RUN] Hard to distinguish graphs
def gen_hard_graphs_WL():

  x1 = torch.ones((10,1))
  edge_index1 = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                 [2, 5, 1, 3, 2, 4, 6, 3, 5, 1, 4, 3, 7, 10, 6, 8, 7, 9, 8, 10, 6, 9]])-1
  y1 = torch.tensor([1])

  x2 = torch.ones((10,1))
  edge_index2 = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                 [2, 6, 1, 3, 7, 2, 4, 10, 3, 5, 4, 6, 1, 5, 2, 8, 7, 9, 8, 10, 3, 9]])-1
  y2 =  torch.tensor([2])

  graph1 = Graph(x=x1, edge_index=edge_index1, y=y1)
  graph2 = Graph(x=x2, edge_index=edge_index2, y=y2)
  return [graph1, graph2]

def hash_node_embedings(node_emb):
  """
  This function is a basic, non-bijective one for visualising the embedings.
  Please use it for guidance, not as a mathematical proof in Part 3.
  It is used just for educational/visualisation purpose.
  You are free to change it with whatever suits you best.
  Hash the tensor representing nodes' features
  to a number in [0,1] used to represent a color

  Args:
    node_emb: list of num_graphs arrays, each of dim (num_nodes x num_feats)
  Returns:
    list of num_graphs arrays in [0,1], each of dim (num_nodes)
  """
  chunk_size_graph = [x.shape[0] for x in node_emb]
  start_idx_graph = [0] + list(itertools.accumulate(chunk_size_graph))[:-1]

  node_emb_flatten = np.concatenate(node_emb).mean(-1)

  min_emb = node_emb_flatten.min()
  max_emb = node_emb_flatten.max()
  node_emb_flatten = (node_emb_flatten-min_emb)/(max_emb-min_emb+0.00001)

  #split in graphs again according to (start_idx_graph, chunk_size_graph)
  node_emb_hashed = [node_emb_flatten[i:i+l] for (i,l) in zip(start_idx_graph, chunk_size_graph)]
  return node_emb_hashed



def main():

    hard_graphs = gen_hard_graphs_WL()
    gallery(hard_graphs, labels=["A", "B"], max_fig_size=(10, 5))

    # Let's try to encode these graphs using our GIN Neural Network.
    BATCH_SIZE = 128  # @param {type:"integer"}
    NUM_EPOCHS = 30  # @param {type:"integer"}
    HIDDEN_DIM = 64  # @param {type:"integer"}
    LR = 0.001  # @param {type:"number"}

    train_zinc_dataset = ZINC(root='', split='train', subset=True)
    val_zinc_dataset = ZINC(root='', split='val', subset=True)
    test_zinc_dataset = ZINC(root='', split='test', subset=True)
    batch_zinc = create_mini_batch(train_zinc_dataset[:3])
    model_simple_gin = SimpleGIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)

    hard_batch = create_mini_batch(hard_graphs)
    out, node_emb = model_simple_gin(hard_batch)

    # split node_emb from batch into separate graphs
    node_emb = node_emb.detach().numpy()
    node_emb_split = [node_emb[:hard_graphs[0].num_nodes], node_emb[hard_graphs[0].num_nodes:]]

    # encode node representation into an int in [0,1] denoting the color
    node_emb_split = hash_node_embedings(node_emb_split)

    gallery(hard_graphs, node_emb=node_emb_split, max_fig_size=(10, 5))

    pass


if __name__ == '__main__':

    main()

    pass