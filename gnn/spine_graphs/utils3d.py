import numpy as np
import torch


def calc_line_point_distance(point_on_line, vector, point):
    """
    Calculate distance between a line and a point.
    The line is defined by a point_on_line and a vector:
        line = point_on_line + vector * t

    Based on:
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    section 'Vector formulation'
    """

    vector = vector / torch.linalg.norm(vector, axis=1, keepdim=True)  # convert to unit vector
    # vector_from_line_to_point = (point - point_on_line) - np.dot((point - point_on_line), vector) * vector
    vector_from_line_to_point = (point - point_on_line) - ((point - point_on_line) * vector).sum(axis=1, keepdim=True) * vector  # implement per row dot product
    distance = torch.linalg.norm(vector_from_line_to_point, axis=1, keepdim=True)

    return distance, vector_from_line_to_point


def calc_point_point_distance(p1, p2, sign=False):

    distance_vector = p2 - p1  # from p1 to p2
    distance = torch.linalg.norm(distance_vector, axis=1, keepdim=True)

    if sign:
        # sign is positive if distance_vector points to the right half space
        # distance *= torch.sign(torch.dot(distance_vector, torch.Tensor([1, 0])))
        distance *= torch.sign(distance_vector @ torch.Tensor([[1, 0]]).T)

    return distance, distance_vector


def calc_angle_between_vectors(v1, v2, units='deg'):

    assert units in ['deg', 'rad']

    # calculate (acute) angle between vectors
    # cos_angle = v1 @ v2.T / (torch.linalg.norm(v1, axis=1) * torch.linalg.norm(v2, axis=1))  # normalized dot product
    cos_angle = (v1 * v2).sum(axis=1, keepdim=True) / (torch.linalg.norm(v1, axis=1, keepdim=True) * torch.linalg.norm(v2, axis=1, keepdim=True))  # normalized dot product
    angle = torch.arccos(cos_angle)

    if units == 'deg':
        angle = torch.rad2deg(angle)

    return angle


