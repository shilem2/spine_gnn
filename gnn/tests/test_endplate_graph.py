
from gnn.spine_graphs.endplate_graph import EndplateGraph

from mid.data import Annotation
from mid.tests import read_test_data


def test_endplate_graph_init():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True)
    keys_sorted = Annotation.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    # ann = Annotation(ann_dict, pixel_spacing, units)
    # ann.plot_annotations()

    graph = EndplateGraph(ann_dict, display=True)

    # TODO: add assertions!


    pass



if __name__ == '__main__':

    test_endplate_graph_init()

    pass