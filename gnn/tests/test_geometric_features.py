import numpy as np
from pytest import approx

from gnn.spine_graphs.geometric_features import calc_spondy, calc_disc_height


def test_calc_spondy():

    upper = np.array([0, 1, 1, 1])
    lower = np.array([-0.5, 0, 0.5, 0])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(-0.5)
    assert spondy_vector == approx(np.array([-0.5, 0]))


    upper = np.array([0, 1, 1, 2])
    lower = np.array([0, 0, 1, 1])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(-np.sqrt(2) / 2)
    assert spondy_vector == approx(np.array([-0.5, -0.5]))


    upper = np.array([0, 1, 1, 0])
    lower = np.array([0, 0, 1, -1])

    spondy, spondy_vector = calc_spondy(upper, lower)

    assert spondy == approx(np.sqrt(2) / 2)
    assert spondy_vector == approx(np.array([0.5, -0.5]))

    pass


def test_disc_height():

    upper = np.array([0, 1, 1, 1])
    lower = np.array([-0.5, 0, 0.5, 0])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(1)
    assert height_vector_lower_upper == approx(np.array([0, 1]))


    upper = np.array([0, 1, 1, 2])
    lower = np.array([0, 0, 1, 1])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(np.sqrt(2) / 2)
    assert height_vector_lower_upper == approx(np.array([-0.5, 0.5]))


    upper = np.array([0, 1, 1, 0])
    lower = np.array([0, 0, 1, -1])

    height, height_vector_lower_upper = calc_disc_height(upper, lower)

    assert height == approx(np.sqrt(2) / 2)
    assert height_vector_lower_upper == approx(np.array([0.5, 0.5]))

    pass


if __name__ == '__main__':

    # test_calc_spondy()
    test_disc_height()

    pass