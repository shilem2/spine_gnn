import numpy as np
from pytest import approx

from gnn.spine_graphs.utils3d import calc_angle_between_vectors, calc_line_point_distance, calc_point_point_distance


def test_angle_between_vectors():

    v1 = np.array([0, 1])
    v2 = np.array([1, 1])
    angle = calc_angle_between_vectors(v1, v2, units='deg')
    assert angle == approx(45)
    angle = calc_angle_between_vectors(v2, v1, units='deg')
    assert angle == approx(45)
    angle = calc_angle_between_vectors(v1, v2, units='rad')
    assert angle == approx(np.pi / 4)

    v3 = np.array([0, -1])
    angle = calc_angle_between_vectors(v1, v3, units='deg')
    assert angle == approx(180)

    v4 = np.array([-1, -1])
    angle = calc_angle_between_vectors(v1, v4, units='deg')
    assert angle == approx(135)


    pass


def test_line_point_distance():

    point_on_line = np.array([1, 0])
    vector = np.array([0, 1])
    point = np.array([0, 1])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(1)
    assert vector_from_line_to_point == approx(np.array([-1, 0]))

    point_on_line = point

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0)
    assert vector_from_line_to_point == approx(np.array([0, 0]))

    point_on_line = np.array([0, 0])
    vector = np.array([1, 1])
    point = np.array([1, 1])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0)
    assert vector_from_line_to_point == approx(np.array([0, 0]))

    point_on_line = np.array([1, 0])
    vector = np.array([-1, 1])
    point = np.array([1, 0])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(0)
    assert vector_from_line_to_point == approx(np.array([0, 0]))


    point_on_line = np.array([1, 0])
    vector = np.array([-1, 1])
    point = np.array([1, 1])

    distance, vector_from_line_to_point = calc_line_point_distance(point_on_line, vector, point)

    assert distance == approx(np.sqrt(2) / 2)
    assert vector_from_line_to_point == approx(np.array([0.5, 0.5]))

    pass


def test_point_point_distance():

    p1 = np.asarray([0, 1])
    p2 = np.asarray([1, 0])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(np.sqrt(2))
    assert distance_vector == approx(np.array([-1, 1]))

    p1 = np.asarray([-1, -1])
    p2 = np.asarray([1, 1])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(np.sqrt(8))
    assert distance_vector == approx(np.array([2, 2]))

    p1 = np.asarray([0, 1])
    p2 = np.asarray([0, 1])

    distance, distance_vector = calc_point_point_distance(p1, p2)

    assert distance == approx(0)
    assert distance_vector == approx(np.array([0, 0]))

    pass

if __name__ == '__main__':

    # test_angle_between_vectors()
    # test_line_point_distance()
    test_point_point_distance()

    pass