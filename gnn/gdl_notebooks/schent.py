
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from torch_geometric.datasets import QM9

from gnn.gdl_notebooks.utils import get_qm9_data
from gnn.gdl_notebooks.train import run_experiment, eval

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def eval_pretrained():
    """Code adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_schnet.py """

    # load QM9 with dense graphs
    train_dataset, val_dataset, test_dataset, std = get_qm9_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    path = './qm9'
    target = 0
    dataset = QM9(path)

    model, datasets = SchNet.from_qm9_pretrained(path, dataset, target)

    model_name = type(model).__name__

    device = 'cpu'
    train_error = eval(model, train_loader, device, std)
    val_error = eval(model, val_loader, device, std)
    test_error = eval(model, test_loader, device, std)

    print(f"\nDone! Train MAE: {train_error:.7f}, Validation MAE: {val_error:.7f}, corresponding test MAE: {test_error:.7f}.")


    # best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    #     model,
    #     model_name,  # "MPNN w/ Features and Coordinates (Invariant Layers)",
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     n_epochs=100
    # )

    pass


def main():

    # load QM9 with dense graphs
    train_dataset, val_dataset, test_dataset, std = get_qm9_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model = FinalMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, coord_dim=3, out_dim=1)
    model = SchNet()

    model_name = type(model).__name__
    best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
        model,
        model_name,  # "MPNN w/ Features and Coordinates (Invariant Layers)",
        train_loader,
        val_loader,
        test_loader,
        n_epochs=100
    )

    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model"])

    RESULTS[model_name] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Test MAE", "Val MAE", "Epoch", "Model"])
    DF_RESULTS = pd.concat((DF_RESULTS, df_temp), ignore_index=True)

    # print(RESULTS)

    sns.set_style('darkgrid')

    fig, ax = plt.subplots()
    p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 2))

    fig, ax = plt.subplots()
    p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))


    pass



if __name__ == '__main__':

    main()
    # eval_pretrained()

    pass