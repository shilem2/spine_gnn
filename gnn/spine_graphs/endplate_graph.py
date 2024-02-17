
from mid.data import Annotation

from gnn.spine_graphs.utils import get_one_hot_dict


class EndplateGraph():

    def __init__(self, ann_dict):

        self.ann_dict = self.sort_ann_dict(ann_dict)
        self.vert_names = list(self.ann_dict.keys())
        self.id2endplate, self.endplate2id = self.get_endplate_dict(self.vert_names)
        self.set_one_hot_dicts()
        self.calc_edge_indices()
        self.calc_edge_features()
        self.calc_node_features()

        # export to graph to pyg.Data
        # display graph using networkx

        pass

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
        self.endplate_one_hot_dict = get_one_hot_dict(self.id2endplate)

        # edge type (same / different vert)
        id2edge_dict = {0: 'vert',  # between upper and lower endplate of the same vert
                        1: 'disc',  # between upper and lower endplate of different verts
                        }
        self.edge_one_hot_dict = get_one_hot_dict(id2edge_dict)

        pass

    def calc_edge_indices(self):

        pass

    def calc_node_features(self):

        pass

    def calc_edge_features(self):

        pass




