from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mid.data import MaccbiDataset

from gnn.spine_graphs import EndplateGraph, EndplateDataset

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def generate_endplate_dataset(n_max=None, s1_upper_only=False, projection='LT'):

    # n_max = None
    # projection = 'LT'
    # s1_upper_only = True
    # s1_upper_only = False

    data_path = '/Users/shilem2/OneDrive - Medtronic PLC/data/2023-01-17_merged_data_v2/'

    data_path = Path(data_path)
    vert_file = data_path / 'vert' / 'vert.parquet'
    rod_file = data_path / 'rod' / 'rod.parquet'
    screw_file = data_path / 'screw' / 'screw.parquet'
    femur_file = data_path / 'femur' / 'femur.parquet'
    dicom_file = data_path / 'dicom' / 'dicom.parquet'

    cfg_update = {}

    dataset = MaccbiDataset(vert_file=vert_file, dicom_file=dicom_file, cfg_update=cfg_update)

    study_id_list = dataset.get_study_id_list()

    graph_list = []
    n = 0
    for study_id in tqdm(study_id_list, desc='study id'):

        study_df = dataset.filter_study_id(study_id, key='dicom_df', projection=projection)
        for index, row in study_df.iterrows():

            try:
                ann = dataset.get_ann(study_id=study_id, projection=row['projection'], body_pos=row['bodyPos'], acquired=row['acquired'],
                                  relative_file_path=row['relative_file_path'], flipped_anns=row['x_sign'], units='mm', display=False, save_fig_name=None)

                ann_dict = ann.values_dict(order='xy', units='mm', vert_anns=True, s1_upper_only=False)
                keys_sorted = ann.sort_keys_by_vert_names(ann_dict.keys())
                ann_dict = {key: ann_dict[key] for key in keys_sorted}

                graph = EndplateGraph(ann_dict, display=False)
                graph_list.append(graph.pyg_graph)
                n += 1

            except:
                pass

            if (n_max is not None) and (n > n_max):
                break  # break inner loop

            pass

        if (n_max is not None) and (n > n_max):
            break  # break outer loop

        pass

    return graph_list


def check_import_timeing():
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
    from gnn.spine_graphs.geometric_features import calc_spondy, calc_disc_height, get_endplate_geometric_data, \
        check_if_lordotic
    from gnn.spine_graphs.utils3d import calc_angle_between_vectors
    from gnn.scripts.generate_dataset import generate_endplate_dataset

    from scipy.stats import ortho_group

    print("PyTorch version {}".format(torch.__version__))
    print("PyG version {}".format(torch_geometric.__version__))

    pass


def check_data():

    import torch
    from torch_geometric.loader import DataLoader
    from gnn.spine_graphs.train_playground import InvariantEndplateMPNNModel

    # some endplate graphs produces NaNs, here we'll investigate it

    dataset = generate_endplate_dataset(n_max=120, s1_upper_only=False, projection='LT')
    dataloader = DataLoader(dataset, batch_size=120, shuffle=True)

    model = InvariantEndplateMPNNModel(num_layers=5, emb_dim=64, in_dim=1, edge_dim=2, out_dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice: {}'.format(device))

    for data in dataloader:
        data = data.to(device)
        y_pred = model(data)
        pass

    pass


def generate_pyg_dataset():

    dataset = EndplateDataset()

    dataset[:10]

    pass


if __name__ == '__main__':

    # generate_endplate_dataset()
    # check_import_timeing()
    # check_data()
    generate_pyg_dataset()


    pass