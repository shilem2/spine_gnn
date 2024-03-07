from pytest import approx
import torch
from gnn.spine_graphs.endplate_graph import EndplateGraph

from mid.data import Annotation
from mid.tests import read_test_data


def test_endplate_graph_init():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    display = False

    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True, s1_upper_only=True)
    keys_sorted = ann1.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    if display:
        ann = Annotation(ann_dict, pixel_spacing, units, dicom_path=img1)
        ann.plot_annotations()

    graph = EndplateGraph(ann_dict, display=display)

    # TODO: add assertions!
    assert set(graph.vert_names) == {'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'}
    assert set(graph.id2endplate.keys()) == {36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50}
    assert set(graph.endplate2id.keys()) == {'T12_upper', 'T12_lower', 'L1_upper', 'L1_lower', 'L2_upper', 'L2_lower', 'L3_upper', 'L3_lower', 'L4_upper', 'L4_lower', 'L5_upper', 'L5_lower', 'S1_upper'}
    assert set(graph.running_index2id.keys()) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    assert graph.edge_index == approx(torch.tensor([[ 0.,  1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.,  5.,  5.,  6.,  6.,  7., 7.,  8.,  8.,  9.,  9., 10., 10., 11., 11., 12.],
                                                    [ 1.,  0.,  2.,  1.,  3.,  2.,  4.,  3.,  5.,  4.,  6.,  5.,  7.,  6., 8.,  7.,  9.,  8., 10.,  9., 11., 10., 12., 11.]]))
    assert graph.edge_features.T == approx(torch.tensor([[1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                                                         [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.]]))
    assert graph.node_features.T == approx(torch.tensor([[36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 50.]]))
    assert graph.node_positions == approx(torch.tensor([[ -96.8160, -612.9752,  117.7045, -563.1919],
                                                        [-127.5910, -448.2384,  101.4120, -420.1787],
                                                        [-135.7370, -412.0324,   95.3290, -379.0879],
                                                        [-142.3260, -248.5994,  109.3055, -242.2633],
                                                        [-134.9980, -202.3274,  105.3405, -204.4521],
                                                        [-125.0870,  -44.9883,  110.2900,  -54.8995],
                                                        [ -95.9880,   -9.3373,  131.3205,  -21.1043],
                                                        [ -77.5130,  116.0396,  154.8785,  110.7922],
                                                        [ -58.1800,  152.0016,  171.1945,  154.6591],
                                                        [ -66.0310,  284.2596,  165.3620,  311.7054],
                                                        [ -91.2810,  310.6057,  116.2090,  398.4321],
                                                        [-138.4880,  442.3456,   56.9255,  522.4872],
                                                        [-157.1510,  478.5757,   11.9150,  581.7716]]), abs=1e-4)
    assert graph.target == approx(torch.tensor([23.3000]))


    pass



if __name__ == '__main__':

    test_endplate_graph_init()

    pass