from pytest import approx
import torch

from gnn.spine_graphs import EndplateGraph, EndplateDataset
from gnn.spine_graphs.geometric_features import calc_spondy, calc_disc_height, get_endplate_geometric_data, check_if_lordotic
from gnn.spine_graphs.utils3d import calc_angle_between_vectors
from gnn.spine_graphs.utils import seed

from gnn.spine_graphs.train_playground import InvariantEndplateMPNNLayer

from mid.data import Annotation
from mid.tests import read_test_data


torch.set_printoptions(sci_mode=False)
seed(42)


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


def test_graph_geometric_feature():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    display = False

    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True, s1_upper_only=True)
    keys_sorted = ann1.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    if display:
        ann = Annotation(ann_dict, pixel_spacing, units, dicom_path=img1)
        ann.plot_annotations()

    graph = EndplateGraph(ann_dict, display=display)

    # calculate geometric features on some nodes
    data = graph.pyg_graph
    # define pos_i, pos_j similar to what happens inside InvariantEndplateMPNNLayer forward
    i = 1  # folo
    j = 0
    edge_index_i = data.edge_index[i]
    edge_index_j = data.edge_index[j]
    pos_i = data.pos.index_select(-2, edge_index_i)
    pos_j = data.pos.index_select(-2, edge_index_j)

    start_i, end_i, distance_i, vector_i, unit_vector_i = get_endplate_geometric_data(pos_i)
    start_j, end_j, distance_j, vector_j, unit_vector_j = get_endplate_geometric_data(pos_j)

    angle = calc_angle_between_vectors(unit_vector_i, unit_vector_j, units='deg')
    is_lordotic = check_if_lordotic(pos_i, pos_j)

    spondy_signed, spondy_vector = calc_spondy(pos_i, pos_j)
    height, height_vector_lower_upper = calc_disc_height(pos_i, pos_j)

    geometric_features = torch.cat([distance_i, unit_vector_i, distance_j, unit_vector_j, angle, is_lordotic, spondy_signed, height], dim=1)

    # reference values were taken from forward run in test_forward_pass_InvariantEndplateMPNNLayer()
    assert pos_i == approx(torch.tensor([[-127.5910, -448.2384,  101.4120, -420.1787],
                                         [ -96.8160, -612.9752,  117.7045, -563.1919],
                                         [-135.7370, -412.0324,   95.3290, -379.0879],
                                         [-127.5910, -448.2384,  101.4120, -420.1787],
                                         [-142.3260, -248.5994,  109.3055, -242.2633],
                                         [-135.7370, -412.0324,   95.3290, -379.0879],
                                         [-134.9980, -202.3274,  105.3405, -204.4521],
                                         [-142.3260, -248.5994,  109.3055, -242.2633],
                                         [-125.0870,  -44.9883,  110.2900,  -54.8995],
                                         [-134.9980, -202.3274,  105.3405, -204.4521],
                                         [ -95.9880,   -9.3373,  131.3205,  -21.1043],
                                         [-125.0870,  -44.9883,  110.2900,  -54.8995],
                                         [ -77.5130,  116.0396,  154.8785,  110.7922],
                                         [ -95.9880,   -9.3373,  131.3205,  -21.1043],
                                         [ -58.1800,  152.0016,  171.1945,  154.6591],
                                         [ -77.5130,  116.0396,  154.8785,  110.7922],
                                         [ -66.0310,  284.2596,  165.3620,  311.7054],
                                         [ -58.1800,  152.0016,  171.1945,  154.6591],
                                         [ -91.2810,  310.6057,  116.2090,  398.4321],
                                         [ -66.0310,  284.2596,  165.3620,  311.7054],
                                         [-138.4880,  442.3456,   56.9255,  522.4872],
                                         [ -91.2810,  310.6057,  116.2090,  398.4321],
                                         [-157.1510,  478.5757,   11.9150,  581.7716],
                                         [-138.4880,  442.3456,   56.9255,  522.4872]]), abs=1e-4)

    assert pos_j == approx(torch.tensor([[ -96.8160, -612.9752,  117.7045, -563.1919],
                                         [-127.5910, -448.2384,  101.4120, -420.1787],
                                         [-127.5910, -448.2384,  101.4120, -420.1787],
                                         [-135.7370, -412.0324,   95.3290, -379.0879],
                                         [-135.7370, -412.0324,   95.3290, -379.0879],
                                         [-142.3260, -248.5994,  109.3055, -242.2633],
                                         [-142.3260, -248.5994,  109.3055, -242.2633],
                                         [-134.9980, -202.3274,  105.3405, -204.4521],
                                         [-134.9980, -202.3274,  105.3405, -204.4521],
                                         [-125.0870,  -44.9883,  110.2900,  -54.8995],
                                         [-125.0870,  -44.9883,  110.2900,  -54.8995],
                                         [ -95.9880,   -9.3373,  131.3205,  -21.1043],
                                         [ -95.9880,   -9.3373,  131.3205,  -21.1043],
                                         [ -77.5130,  116.0396,  154.8785,  110.7922],
                                         [ -77.5130,  116.0396,  154.8785,  110.7922],
                                         [ -58.1800,  152.0016,  171.1945,  154.6591],
                                         [ -58.1800,  152.0016,  171.1945,  154.6591],
                                         [ -66.0310,  284.2596,  165.3620,  311.7054],
                                         [ -66.0310,  284.2596,  165.3620,  311.7054],
                                         [ -91.2810,  310.6057,  116.2090,  398.4321],
                                         [ -91.2810,  310.6057,  116.2090,  398.4321],
                                         [-138.4880,  442.3456,   56.9255,  522.4872],
                                         [-138.4880,  442.3456,   56.9255,  522.4872],
                                         [-157.1510,  478.5757,   11.9150,  581.7716]]), abs=1e-4)

    assert geometric_features == approx(torch.tensor([[   230.7157,      0.9926,      0.1216,    220.2213,      0.9741,
                                                          0.2261,      6.0796,     -1.0000,     -7.2621,    167.4293],
                                                      [   220.2213,      0.9741,      0.2261,    230.7157,      0.9926,
                                                          0.1216,      6.0796,     -1.0000,    -10.5113,    167.2568],
                                                      [   233.4027,      0.9900,      0.1411,    230.7157,      0.9926,
                                                          0.1216,      1.1287,      1.0000,      3.6821,     36.9279],
                                                      [   230.7157,      0.9926,      0.1216,    233.4027,      0.9900,
                                                          0.1411,      1.1287,      1.0000,     -2.9540,     36.9933],
                                                      [   251.7112,      0.9997,      0.0252,    233.4027,      0.9900,
                                                          0.1411,      6.6719,     -1.0000,    -16.5454,    162.7268],
                                                      [   233.4027,      0.9900,      0.1411,    251.7112,      0.9997,
                                                          0.0252,      6.6719,     -1.0000,     -2.4730,    163.5471],
                                                      [   240.3479,      1.0000,     -0.0088,    251.7112,      0.9997,
                                                          0.0252,      1.9488,     -1.0000,     -8.4904,     46.0729],
                                                      [   251.7112,      0.9997,      0.0252,    240.3479,      1.0000,
                                                          -0.0088,      1.9488,     -1.0000,      6.9187,     46.3350],
                                                      [   235.5855,      0.9991,     -0.0421,    240.3479,      1.0000,
                                                          -0.0088,      1.9044,     -1.0000,     -8.5197,    157.4205],
                                                      [   240.3479,      1.0000,     -0.0088,    235.5855,      0.9991,
                                                          -0.0421,      1.9044,     -1.0000,      3.2829,    157.6167],
                                                      [   227.6128,      0.9987,     -0.0517,    235.5855,      0.9991,
                                                          -0.0421,      0.5518,     -1.0000,    -27.5734,     36.8436],
                                                      [   235.5855,      0.9991,     -0.0421,    227.6128,      0.9987,
                                                          -0.0517,      0.5518,     -1.0000,     27.2170,     37.1077],
                                                      [   232.4507,      0.9997,     -0.0226,    227.6128,      0.9987,
                                                          -0.0517,      1.6696,      1.0000,    -11.9686,    126.1644],
                                                      [   227.6128,      0.9987,     -0.0517,    232.4507,      0.9997,
                                                          -0.0226,      1.6696,      1.0000,     15.6400,    125.7621],
                                                      [   229.3899,      0.9999,      0.0116,    232.4507,      0.9997,
                                                          -0.0226,      1.9573,      1.0000,    -18.5163,     36.3893],
                                                      [   232.4507,      0.9997,     -0.0226,    229.3899,      0.9999,
                                                          0.0116,      1.9573,      1.0000,     19.7483,     35.7356],
                                                      [   233.0150,      0.9930,      0.1178,    229.3899,      0.9999,
                                                          0.0116,      6.1005,      1.0000,      6.3183,    132.3401],
                                                      [   229.3899,      0.9999,      0.0116,    233.0150,      0.9930,
                                                          0.1178,      6.1005,      1.0000,      7.7817,    132.2621],
                                                      [   225.3122,      0.9209,      0.3898,    233.0150,      0.9930,
                                                          0.1178,     16.1777,      1.0000,     21.9711,     29.1367],
                                                      [   233.0150,      0.9930,      0.1178,    225.3122,      0.9209,
                                                          0.3898,     16.1777,      1.0000,    -12.9831,     34.1045],
                                                      [   211.2086,      0.9252,      0.3794,    225.3122,      0.9209,
                                                          0.3898,      0.6429,     -1.0000,     -7.8792,    139.7205],
                                                      [   225.3122,      0.9209,      0.3898,    211.2086,      0.9252,
                                                          0.3794,      0.6429,     -1.0000,      6.3112,    139.8001],
                                                      [   198.0725,      0.8536,      0.5210,    211.2086,      0.9252,
                                                          0.3794,      9.1002,      1.0000,      3.5201,     40.6022],
                                                      [   211.2086,      0.9252,      0.3794,    198.0725,      0.8536,
                                                          0.5210,      9.1002,      1.0000,      2.9460,     40.6479]]), abs=1e-4)

    pass


def test_forward_pass_InvariantEndplateMPNNLayer():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]
    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True, s1_upper_only=True)
    keys_sorted = ann1.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    # calculate geometric features on some nodes
    graph = EndplateGraph(ann_dict)
    data = graph.pyg_graph

    layer = InvariantEndplateMPNNLayer(emb_dim=1, edge_dim=2, geometric_feat_dim=10, aggr='add')

    out = layer.forward(data.x, data.pos, data.edge_index, data.edge_attr)

    assert out.T.detach() == approx(torch.tensor([[0.6647, 0.6647, 0.6647, 0.6647, 0.6647, 0.6647, 0.6647, 0.3818, 0.0463, 0.0000, 0.0000, 0.0000, 0.0000]]), abs=1e-4)

    pass


def test_load_endplate_dataset():

    dataset = EndplateDataset()

    assert len(dataset) == 4033
    assert str(dataset[1001]) == 'Data(x=[17, 1], edge_index=[2, 32], edge_attr=[32, 2], y=[1], pos=[17, 4])'
    assert str(dataset[4001]) == 'Data(x=[13, 1], edge_index=[2, 24], edge_attr=[24, 2], y=[1], pos=[13, 4])'

    pass


if __name__ == '__main__':

    # test_endplate_graph_init()
    # test_graph_geometric_feature()
    # test_forward_pass_InvariantEndplateMPNNLayer()
    test_load_endplate_dataset()

    pass