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


if __name__ == '__main__':

    test_angle_between_vectors()

    pass