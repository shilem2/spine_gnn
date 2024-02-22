import torch
import numpy as np
from pytest import approx

from gnn.spine_graphs.utils3d import calc_angle_between_vectors, calc_line_point_distance, calc_point_point_distance


def test_angle_between_vectors():

    v1 = torch.Tensor([[0, 1]])
    v2 = torch.Tensor([[1, 1]])
    angle = calc_angle_between_vectors(v1, v2, units='deg')
    assert angle == approx(45)
    angle = calc_angle_between_vectors(v2, v1, units='deg')
    assert angle == approx(45)
    angle = calc_angle_between_vectors(v1, v2, units='rad')
    assert angle == approx(torch.pi / 4)

    v3 = torch.Tensor([[0, -1]])
    angle = calc_angle_between_vectors(v1, v3, units='deg')
    assert angle == approx(180)

    v4 = torch.Tensor([[-1, -1]])
    angle = calc_angle_between_vectors(v1, v4, units='deg')
    assert angle == approx(135)


    pass


def test_line_point_distance():

    point_on_line = torch.Tensor([[1, 0]])
    vector = torch.Tensor([[0, 1]])
    point = torch.Tensor([[0, 1]])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(1)
    assert vector_from_line_to_point == approx(torch.Tensor([[-1, 0]]))

    point_on_line = point

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0)
    assert vector_from_line_to_point == approx(torch.Tensor([[0, 0]]))

    point_on_line = torch.Tensor([[0, 0]])
    vector = torch.Tensor([[1, 1]])
    point = torch.Tensor([[1, 1]])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0, abs=1e-7)
    assert vector_from_line_to_point == approx(torch.Tensor([[0, 0]]), abs=1e-7)

    point_on_line = torch.Tensor([[1, 0]])
    vector = torch.Tensor([[-1, 1]])
    point = torch.Tensor([[1, 0]])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0)
    assert vector_from_line_to_point == approx(torch.Tensor([[0, 0]]))


    point_on_line = torch.Tensor([[1, 0]])
    vector = torch.Tensor([[-1, 1]])
    point = torch.Tensor([[1, 1]])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(np.sqrt(2) / 2)
    assert vector_from_line_to_point == approx(torch.Tensor([[0.5, 0.5]]))

    pass


def test_point_point_distance():

    p1 = torch.Tensor([[0, 1]])
    p2 = torch.Tensor([[1, 0]])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(np.sqrt(2))
    assert distance_vector == approx(torch.Tensor([[1, -1]]))

    p1 = torch.Tensor([[-1, -1]])
    p2 = torch.Tensor([[1, 1]])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(np.sqrt(8))
    assert distance_vector == approx(torch.Tensor([[2, 2]]))

    p1 = torch.Tensor([[0, 1]])
    p2 = torch.Tensor([[0, 1]])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(0)
    assert distance_vector == approx(torch.Tensor([[0, 0]]))

    pass



if __name__ == '__main__':

    test_angle_between_vectors()
    test_line_point_distance()
    test_point_point_distance()

    pass