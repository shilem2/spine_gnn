import numpy as np
from pytest import approx

from gnn.spine_graphs.geometric_features import calc_spondy


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


if __name__ == '__main__':

    test_calc_spondy()

    pass