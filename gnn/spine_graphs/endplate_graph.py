import numpy as np
import torch
from torch_geometric.data import Data

from mid.data import Annotation

from gnn.spine_graphs.utils import get_one_hot_dict
from gnn.spine_graphs.visualization_utils import gallery


class EndplateGraph():

    def __init__(self, ann_dict, display=False):

        self.ann_dict = self.sort_ann_dict(ann_dict)
        self.vert_names = list(self.ann_dict.keys())
        self.id2endplate, self.endplate2id = self.get_endplate_dict(self.vert_names)  # global id - starting for C1, ending in S1
        self.id2running_index = {id: n for n, id in enumerate(self.id2endplate.keys())}  # local id, where id 0 is the upper most endplate in current spine
        self.set_one_hot_dicts()
        self.calc_edge_indices()
        self.calc_node_features()
        self.calc_node_positions()
        self.calc_edge_features()


        # TODO:
        # add targets - node / graph ? start with LL?
        # export to graph to pyg.Data
        # display graph using networkx

        self.pyg_graph = self.export_to_pyg_data()

        if display:
            gallery(self.pyg_graph)

        pass

    def export_to_pyg_data(self):

        data = Data(
            x=torch.Tensor(self.node_features),
            edge_index=torch.Tensor(self.edge_index),
            edge_attr=torch.Tensor(self.edge_features),
            pos=torch.Tensor(self.node_positions),
        )

        return data

    def get_endplate_dict(self, vert_names=None, vert_names_global=Annotation.vert_names):
        """
        Get dictionary of endplates names and ids, ordered by canonical order of Annotation.vert_names
        """

        vert_names = vert_names_global if vert_names is None else Annotation.sort_keys_by_vert_names(vert_names)

        id2endplate = {}
        n = 0
        for vert in vert_names_global:
            if vert in vert_names:
                id2endplate[n] = '{}_upper'.format(vert)
                id2endplate[n + 1] = '{}_lower'.format(vert)
            n += 2

        endplate2id = {endplate: id for id, endplate in id2endplate.items()}

        return id2endplate, endplate2id

    def sort_ann_dict(self, ann_dict):
        keys_sorted = Annotation.sort_keys_by_vert_names(ann_dict.keys())
        ann_dict = {key: ann_dict[key] for key in keys_sorted}
        return ann_dict

    def set_one_hot_dicts(self):

        # endplate type (L1_upper, L2_lower, ...)
        self.endplate_one_hot_dict = get_one_hot_dict(self.get_endplate_dict()[0])  # call get_endplate_dict() without arguments, to create dict using global endplate order

        # edge type (same / different vert)
        id2edge_dict = {0: 'vert',  # between upper and lower endplate of the same vert
                        1: 'disc',  # between upper and lower endplate of different verts
                        }
        self.edge_one_hot_dict = get_one_hot_dict(id2edge_dict)

        pass

    def calc_edge_indices(self):

        # iterate over endplates and compute edge indices
        edge_index = []
        for id, endplate in self.id2endplate.items():

            if (id + 1) in self.id2endplate:
                # edge index - enter both ways to create undirected graph
                ei = [[self.id2running_index[id], self.id2running_index[id + 1]],
                      [self.id2running_index[id + 1], self.id2running_index[id]]]
                edge_index.extend(ei)
            pass

        self.edge_index = np.asarray(edge_index).transpose()

        pass

    def calc_node_features(self):

        node_features = []
        for id, endplate in self.id2endplate.items():

            nf = [self.endplate_one_hot_dict[id]]
            node_features.extend(nf)

            pass

        self.node_features = np.asarray(node_features)

        pass

    def calc_node_positions(self):

        """
        Get endplate position coordinates from ann_dict.

        Each item in ann_dict is structured as follows
                     pos = nparray(['upperVertSt_x', 'upperVertSt_y',   # TL
                                   'upperVertEnd_x', 'upperVertEnd_y',  # TR
                                   'lowerVertEnd_x', 'lowerVertEnd_y',  # BR
                                   'lowerVertSt_x', 'lowerVertSt_y',    # BL
                                   ])
        """

        # calc mean position as coordinate system start
        values = np.concatenate([val for key, val in self.ann_dict.items()])  # xy, original units
        pos_mean = values.mean(axis=0)

        node_positions = []
        for id, endplate in self.id2endplate.items():
            vert_name, endplate_type = endplate.split('_')
            vert_position = self.ann_dict[vert_name]
            start_index = 0 if endplate_type == 'upper' else 2
            endplate_position = vert_position[start_index:start_index+2, :] - pos_mean
            node_position = [endplate_position.reshape(1, 4)]
            node_positions.extend(node_position)

            pass

        self.node_positions = np.concatenate(node_positions, axis=0)

        pass

    def calc_edge_features(self):

        edge_feature = []
        for id, endplate in self.id2endplate.items():

            if (id + 1) in self.id2endplate:
                edge_id = 0 if endplate.split('_')[0] == self.id2endplate[id + 1].split('_')[0] else 1  # 0 - same vert, 1 - differet vert (disc)
                ef = [self.edge_one_hot_dict[edge_id], self.edge_one_hot_dict[edge_id]]  # enter twice = undirected graph

                # TODO: add geometric features calculations: angle, height, spondy

                edge_feature.extend(ef)
            pass

        self.edge_features = np.asarray(edge_feature)

        pass




