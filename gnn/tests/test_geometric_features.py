import torch
import numpy as np
from pytest import approx

from gnn.spine_graphs.geometric_features import calc_spondy, calc_disc_height, calc_lumbar_lordosis_angle
from gnn.spine_graphs.endplate_graph import EndplateGraph

from mid.data import Annotation
from mid.tests import read_test_data


def test_calc_spondy():

    upper = torch.Tensor([[0, 1, 1, 1]])
    lower = torch.Tensor([[-0.5, 0, 0.5, 0]])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(-0.5)
    assert spondy_vector == approx(torch.Tensor([[-0.5, 0]]))


    upper = torch.Tensor([[0, 1, 1, 2]])
    lower = torch.Tensor([[0, 0, 1, 1]])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(-np.sqrt(2) / 2)
    assert spondy_vector == approx(torch.Tensor([[-0.5, -0.5]]))


    upper = torch.Tensor([[0, 1, 1, 0]])
    lower = torch.Tensor([[0, 0, 1, -1]])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(np.sqrt(2) / 2)
    assert spondy_vector == approx(torch.Tensor([[0.5, -0.5]]))

    pass


def test_disc_height():

    upper = torch.Tensor([[0, 1, 1, 1]])
    lower = torch.Tensor([[-0.5, 0, 0.5, 0]])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(1)
    assert height_vector_lower_upper == approx(torch.Tensor([[0, 1]]))


    upper = torch.Tensor([[0, 1, 1, 2]])
    lower = torch.Tensor([[0, 0, 1, 1]])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(np.sqrt(2) / 2)
    assert height_vector_lower_upper == approx(torch.Tensor([[-0.5, 0.5]]))


    upper = torch.Tensor([[0, 1, 1, 0]])
    lower = torch.Tensor([[0, 0, 1, -1]])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(np.sqrt(2) / 2)
    assert height_vector_lower_upper == approx(torch.Tensor([[0.5, 0.5]]))

    pass

def test_lumbar_lordosis():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True)
    keys_sorted = Annotation.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    # ann = Annotation(ann_dict, pixel_spacing, units)
    # ann.plot_annotations()

    graph = EndplateGraph(ann_dict, display=True)

    LL_angle, is_lordotic = calc_lumbar_lordosis_angle(graph, units='deg')

    assert LL_angle == approx(23.3)
    assert is_lordotic == approx(1)

    pass


if __name__ == '__main__':

    test_calc_spondy()
    test_disc_height()
    test_lumbar_lordosis()

    pass