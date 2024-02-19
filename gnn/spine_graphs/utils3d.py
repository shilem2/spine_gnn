import numpy as np


def calc_line_point_distance(point_on_line, vector, point):
    """
    Calculate distance between a line and a point.
    The line is defined by a point_on_line and a vector:
        line = point_on_line + vector * t

    Based on:
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    section 'Vector formulation'
    """

    vector = vector / np.linalg.norm(vector)  # convert to unit vector
    vector_from_line_to_point = (point - point_on_line) - np.dot((point - point_on_line), vector) * vector
    distance = np.linalg.norm(vector_from_line_to_point)

    return distance, vector_from_line_to_point


def calc_point_point_distance(point1, point2, sign=True):

    distance_vector = point2 - point1
    distance = np.linalg.norm(distance_vector)

    if sign:
        distance *= np.sign(distance_vector)

    return distance, distance_vector


def calc_angle_between_vectors(v1, v2, units='deg'):

    assert units in ['deg', 'rad']

    # calculate (acute) angle between vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    if units == 'deg':
        angle = np.rad2deg(angle)

    return angle

