from pathlib import Path
from tqdm import tqdm
import shutil

from torch_geometric.data import InMemoryDataset

from gnn.spine_graphs import EndplateGraph

from mid.data import MaccbiDataset


class EndplateDataset(InMemoryDataset):

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):

        if root is None:
            root = (Path(__file__).parents[2] / 'data' / 'endplate').as_posix()

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

        pass

    @property
    def raw_file_names(self):
        return ['vert.parquet', 'dicom.parquet']

    @property
    def processed_file_names(self):
        return ['endplate_dataset.p']

    def download(self, data_path='/Users/shilem2/OneDrive - Medtronic PLC/data/2023-01-17_merged_data_v2/'):

        vert_df_file = Path(data_path) / 'vert' / 'vert.parquet'
        dicom_df_file = Path(data_path) / 'dicom' / 'dicom.parquet'
        shutil.copy(vert_df_file, Path(self.root) / 'raw')
        shutil.copy(dicom_df_file, Path(self.root) / 'raw')

        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = generate_endplate_graph_list(self.root, n_max=None, s1_upper_only=False, projection='LT')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

        pass

def generate_endplate_graph_list(data_path, n_max=None, s1_upper_only=False, projection='LT'):

    data_path = Path(data_path)
    vert_file = data_path / 'raw' /'vert.parquet'
    # rod_file = data_path / 'raw' / 'rod.parquet'
    # screw_file = data_path / 'raw' / 'screw.parquet'
    # femur_file = data_path / 'raw' / 'femur.parquet'
    dicom_file = data_path / 'raw' / 'dicom.parquet'

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

                graph = EndplateGraph(ann_dict, endplate_feature_type='ordinal', target_type='LL', s1_upper_only=True, display=False)
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
