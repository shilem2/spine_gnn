
import torch

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from gnn.gdl_notebooks.utils import get_qm9_data, SetTarget
from gnn.gdl_notebooks.train import run_experiment
from gnn.gdl_notebooks.practical_3_part_1_2_basic_mpnn_3d import MPNNModel, CoordMPNNModel
from gnn.gdl_notebooks.practical_3_part_3_mpnn_3d_rot_trans_invariant import InvariantMPNNModel
from gnn.gdl_notebooks.practical_3_part_4_equivariant_model import FinalMPNNModel

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    # load QM9 with dense graphs
    train_dataset, val_dataset, test_dataset, std = get_qm9_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Load QM9 dataset with sparse graphs (by removing the full graphs transform)
    path = './qm9'
    target = 0
    sparse_dataset = QM9(path, transform=SetTarget())

    # Normalize targets per data sample to mean = 0 and std = 1.
    mean = sparse_dataset.data.y.mean(dim=0, keepdim=True)
    std = sparse_dataset.data.y.std(dim=0, keepdim=True)
    sparse_dataset.data.y = (sparse_dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

    # Split datasets (3K subset)
    train_dataset_sparse = sparse_dataset[:1000]
    val_dataset_sparse = sparse_dataset[1000:2000]
    test_dataset_sparse = sparse_dataset[2000:]
    print(f"Created sparse dataset splits with {len(train_dataset_sparse)} training, {len(val_dataset_sparse)} validation, {len(test_dataset_sparse)} test samples.")

    # Create dataloaders with batch size = 32
    train_loader_sparse = DataLoader(train_dataset_sparse, batch_size=32, shuffle=True)
    test_loader_sparse = DataLoader(test_dataset_sparse, batch_size=32, shuffle=False)
    val_loader_sparse = DataLoader(val_dataset_sparse, batch_size=32, shuffle=False)

    val_batch_sparse = next(iter(val_loader_sparse))
    val_batch_dense = next(iter(val_loader))

    # These two batches should correspond to the same molecules. Let's add a sanity check
    assert torch.allclose(val_batch_sparse.y, val_batch_dense.y, atol=1e-4)

    print(f"Number of edges in sparse batch {val_batch_sparse.edge_index.shape[-1]}. Number of edges in dense batch {val_batch_dense.edge_index.shape[-1]}")

    sparse_results = {}
    # dense_results = RESULTS
    dense_results = {}

    # ============ YOUR CODE HERE ==============
    # Instantiate your models
    models = [MPNNModel(         num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1),
              CoordMPNNModel(    num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, coord_dim=3),
              InvariantMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1),
              FinalMPNNModel(    num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, coord_dim=3, out_dim=1),
              ]
    # ==========================================

    for model in models:
        model_name = type(model).__name__

        if model_name not in sparse_results:
            sparse_results[model_name] = run_experiment(
                model,
                model_name,
                train_loader_sparse,
                val_loader_sparse,
                test_loader_sparse,
                n_epochs=100
            )

        if model_name not in dense_results:
            dense_results[model_name] = run_experiment(
                model,
                model_name,
                train_loader,
                val_loader,
                test_loader,
                n_epochs=100
            )

    df_sparse = pd.DataFrame.from_dict(sparse_results, orient='index',
                                       columns=['Best val MAE', 'Test MAE', 'Train time', 'Train History'])
    df_dense = pd.DataFrame.from_dict(dense_results, orient='index', columns=['Best val MAE', 'Test MAE', 'Train time'])
    df_sparse['type'] = 'sparse'
    df_dense['type'] = 'dense'
    df = df_sparse.append(df_dense)

    sns.set(rc={'figure.figsize': (10, 6)})
    sns.barplot(x=df.index, y="Test MAE", hue="type", data=df);

    # You might want to save and download this plot
    plt.savefig("comparison.png")
    # files.download("comparison.png")

    pass

if __name__ == '__main__':

    main()

    pass