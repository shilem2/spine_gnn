from pathlib import Path
import pandas as pd

from mid.data import MaccbiDataset

from gnn.spine_graphs.endplate_graph import EndplateGraph

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
    for study_id in study_id_list:

        study_df = dataset.filter_study_id(study_id, key='dicom_df', projection=projection)
        for index, row in study_df.iterrows():

            ann = dataset.get_ann(study_id=study_id, projection=row['projection'], body_pos=row['bodyPos'], acquired=row['acquired'],
                                  relative_file_path=row['relative_file_path'], flipped_anns=row['x_sign'], units='mm', display=False, save_fig_name=None)

            ann_dict = ann.values_dict(order='xy', units='mm', vert_anns=True, s1_upper_only=False)
            keys_sorted = ann.sort_keys_by_vert_names(ann_dict.keys())
            ann_dict = {key: ann_dict[key] for key in keys_sorted}

            try:
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





if __name__ == '__main__':

    generate_endplate_dataset()

    pass