import numpy as np
import torch

from gnn.spine_graphs.utils3d import calc_line_point_distance, calc_point_point_distance, calc_angle_between_vectors


def calc_spondy(upper, lower):
    """
    calculate Spondylolisthesis of lower endplate.
    The calculation is done by downloading a perpendicular line from the back of the upper endplate, to the line defined
    by the lower endplate.

    Parameters
    ----------
    upper : ndarray
        Upper endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.
    lower : ndarray
        lower endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.

    Returns
    -------
    spondy: float
        A scalar signed spondy distance.
        Positive value means that the back of the lower endplate is more forward than the back of the upper endplate.

    spondy_vector : ndarray
        Vector pointing from the perpendicular line and lower endplate intersection point, to the backward part of the
        lower endplate.
    """

    upper_start = upper[:, :2]
    lower_start = lower[:, :2]
    lower_end = lower[:, 2:]
    lower_vector = lower_end - lower_start

    # calculate distance between upper and lower start points
    distance_upper_lower, distance_vector_lower_upper = calc_line_point_distance(lower_start, lower_vector, upper_start)

    # get intersection point between perpendicular and lower
    distance_vector_upper_lower = - distance_vector_lower_upper
    intersection_point = upper_start + distance_vector_upper_lower

    spondy_signed, spondy_vector = calc_point_point_distance(intersection_point, lower_start, sign=True)      # sign is positive if distance_vector points to the right half space

    return spondy_signed, spondy_vector

def calc_disc_height(upper, lower):
    """
    calculate disc height.
    The calculation is done by downloading a perpendicular line from the back of the upper endplate, to the line defined
    by the lower endplate.

    Parameters
    ----------
    upper : ndarray
        Upper endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.
    lower : ndarray
        lower endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.

    Returns
    -------
    height: float
        Disc height.

    height_vector : ndarray
        Vector pointing from the perpendicular line and lower endplate intersection point, to the backward part of the
        upper endplate.
    """

    upper_start = upper[:, :2]
    lower_start = lower[:, :2]
    lower_end = lower[:, 2:]
    lower_vector = lower_end - lower_start

    # calculate distance between upper and lower start points
    height, height_vector_lower_upper = calc_line_point_distance(lower_start, lower_vector, upper_start)

    return height, height_vector_lower_upper


def calc_lumbar_lordosis_angle(graph, units='deg'):
    """
    Calculate lumbar lordosis (LL) angle, and check if it is lordotic or kyphotic.

    To test if the angle is lordotic or kyphotic, we assume that the endplate start point is posterior and the endplate
    end point is anterior.
    Then we check which points are closer together - if the posterior start points are closer than the anterior end
    points, we say the the angle is lordotic, otherwise, way say that it is kyphotic.

    Parameters
    ----------
    graph : EndplateGraph
        Endplate graph object.
    units : str, optional
        Wanted angle units.

    Returns
    -------
    LL_angle: float
        Lumbar lordosis angle.

    is_lordotic : bool
        Boolean flag, True if LL angle is lordotic, False if it is kyphotic.
    """

    L1_upper_start, L1_upper_end, L1_upper_vector = graph.get_endplate_position('L1_upper')
    S1_upper_start, S1_upper_end, S1_upper_vector = graph.get_endplate_position('S1_upper')

    if (L1_upper_vector is not None) and (S1_upper_vector is not None):

        LL_angle = calc_angle_between_vectors(L1_upper_vector, S1_upper_vector, units=units).round(decimals=1)

        # check if angle is lordotic or kyphotic
        posterior_distance = calc_point_point_distance(L1_upper_start, S1_upper_start)[0]
        anterior_distance = calc_point_point_distance(L1_upper_end, S1_upper_end)[0]
        is_lordotic = 1. if posterior_distance <= anterior_distance else -1.

    else:
        LL_angle = None
        is_lordotic = None

    return LL_angle, is_lordotic

def check_if_lordotic(upper, lower):
    """
    Check if an angle is lordotic or kyphotic.

    To test if the angle is lordotic or kyphotic, we assume that the endplate start point is posterior and the endplate
    end point is anterior.
    Then we check which points are closer together - if the posterior start points are closer than the anterior end
    points, we say that the angle is lordotic, otherwise, way say that it is kyphotic.

    Parameters
    ----------
    upper : ndarray
        Upper endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.
    lower : ndarray
        lower endplate, given as a 4 values array [x_start, y_start, x_end, y_end], where start is the backward part of
        the endplate, and end is the forward part.

    Returns
    -------
    is_lordotic: float
        1 if angle is lordotic, -1 if it is kyphotic.
    """

    upper_start, upper_end = get_endplate_geometric_data(upper)[:2]
    lower_start, lower_end = get_endplate_geometric_data(lower)[:2]

    # check if angle is lordotic or kyphotic
    posterior_distance = calc_point_point_distance(upper_start, lower_start)[0]
    anterior_distance = calc_point_point_distance(upper_end, lower_end)[0]
    is_lordotic_flag = posterior_distance <= anterior_distance
    is_lordotic = torch.ones(is_lordotic_flag.shape).fill_(-1)
    is_lordotic[is_lordotic_flag] = 1.

    return is_lordotic


def get_endplate_geometric_data(endplate_position):

    start = endplate_position[:, :2]
    end = endplate_position[:, 2:]
    distance, vector = calc_point_point_distance(start, end)
    unit_vector = vector / distance

    return start, end, distance, vector, unit_vector

