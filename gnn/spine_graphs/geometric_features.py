import numpy as np

from gnn.spine_graphs.utils3d import calc_line_point_distance, calc_point_point_distance


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

    upper_start = upper[:2]
    lower_start = lower[:2]
    lower_end = lower[2:]
    lower_vector = lower_end - lower_start

    # calculate distance between upper and lower start points
    distance_upper_lower, distance_vector_lower_upper = calc_line_point_distance(lower_start, lower_vector, upper_start)

    # get intersection point between perpendicular and lower
    distance_vector_upper_lower = - distance_vector_lower_upper
    intersection_point = upper_start + distance_vector_upper_lower

    spondy_signed, spondy_vector = calc_point_point_distance(intersection_point, lower_start, sign=True)      # sign is positive if distance_vector points to the right half space

    return spondy_signed, spondy_vector