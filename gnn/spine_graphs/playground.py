import numpy as np

from mid.tests import read_test_data, read_data
from mid.data import Annotation, plot_annotations

from gnn.spine_graphs.endplate_graph import EndplateGraph

def generate_simple_spine_graph():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    ann_dict = ann1.values_dict(order='xy', units='mm', vert_anns=True)
    keys_sorted = Annotation.sort_keys_by_vert_names(ann_dict.keys())
    ann_dict = {key: ann_dict[key] for key in keys_sorted}

    ann = Annotation(ann_dict, pixel_spacing, units)

    # ann.plot_annotations()

    # graph = generate_spine_graph(ann_dict, graph_type='endplate')

    graph = EndplateGraph(ann_dict, display=True)


    pass

def generate_spine_graph(ann_dict, graph_type='endplate'):

    assert graph_type in ['keypoint', 'endplate', 'vert']

    if graph_type == 'keypoint':
        graph = generate_spine_keypoint_graph(ann_dict)
    elif graph_type == 'endplate':
        graph = generate_spine_endplate_graph(ann_dict)

    pass

def generate_spine_endplate_graph(ann_dict, display=False):

    edge_index = []
    node_feature = []
    node_position = []
    edge_feature = []
    node_target = []

    # setup
    vert_names = list(ann_dict.keys())
    id2endplate, endplate2id = get_endplate_dict(vert_names)
    endplate_one_hot = generate_endplate_one_hot()
    edge_one_hot = generate_edge_one_hot()
    id2running_index = {id: n for n, id in enumerate(id2endplate.keys())}

    # iterate over endplates and compute of the above - features, indices, etc.
    for id, endplate in id2endplate.items():

        if (id + 1) in id2endplate:
            # edge index
            ei = [[id2running_index[id], id2running_index[id + 1]], [id2running_index[id + 1], id2running_index[id]]]
            edge_index.extend(ei)

            # edge features
            edge_id = 0 if endplate.split('_')[0] == id2endplate[id+1].split('_')[0] else 1  # 0 - same vert, 1 - differet vert (disc)
            ef = [edge_one_hot[edge_id], edge_one_hot[edge_id]]
            edge_feature.extend(ef)

        # node features

        # node position


        pass

    pass

def generate_spine_keypoint_graph(ann_dict, display=False):

    # edge index
    # node features
    # edge features
    # position (coordinates)
    # target ?

    edge_index = []
    node_feature = []
    node_position = []
    edge_feature = []
    node_target = []

    verts = list(ann_dict.keys())
    discs = ['{}_{}'.format(verts[n], verts[n+1]) for n in range(len(verts) - 1)]  # assume verts are sorted in ascending order

    # iterate over verts
    i = j = 0
    for vert in verts:

        #              pos = nparray(['upperVertSt_x', 'upperVertSt_y',   # TL
        #                            'upperVertEnd_x', 'upperVertEnd_y',  # TR
        #                            'lowerVertEnd_x', 'lowerVertEnd_y',  # BR
        #                            'lowerVertSt_x', 'lowerVertSt_y',    # BL
        #                            ])
        # add save vert data

        pos = ann_dict[vert]
        for n in range(pos.shape[0]):
            node_position.append(pos[n, :])
            # node_feature =

        # add adjacent vert data
        # loop over neighbor verts


    if display:
        pass

    pass

def get_global_vert_dict(vert_names=Annotation.vert_names):

    id2vert = {id: vert for id, vert in enumerate(vert_names)}
    vert2id = {vert: id for id, vert in id2vert.items()}

    return id2vert, vert2id

def get_global_disc_dict(vert_names=Annotation.vert_names):

    id2disc = {n: '{}_{}'.format(vert_names[n], vert_names[n+1]) for n in range(len(vert_names) - 1)}  # assume verts are sorted in ascending order
    disc2id = {disc:  id for id, disc in id2disc.items()}

    return id2disc, disc2id

def get_endplate_dict(vert_names=None, vert_names_global=Annotation.vert_names):
    """
    Get dictionary of endplates names and ids, ordered by canonical order of Annotation.vert_names
    """

    vert_names = vert_names_global if vert_names is None else Annotation.sort_keys_by_vert_names(vert_names)

    id2endplate = {}
    n = 0
    for vert in vert_names_global:
        if vert in vert_names:
            id2endplate[n] = '{}_upper'.format(vert)
            id2endplate[n+1] = '{}_lower'.format(vert)
        n += 2

    endplate2id = {endplate: id for id, endplate in id2endplate.items()}

    return id2endplate, endplate2id

# def get_endplate_dict(vert_names):
#
#     vert_names = Annotation.sort_keys_by_vert_names(vert_names)
#
#     id2endplate_global = get_global_endplate_dict()
#
#     # TODO: continue
#
#     return endplate_dict

def get_one_hot(id2data_dict):
    """
    Get one-hot row embedding for each item in id2data_dict.
    """

    ids = list(id2data_dict.keys())
    N = len(ids)
    one_hot_matrix = np.identity(N)
    one_hot_dict = {id: one_hot_matrix[n, :] for n, id in enumerate(ids)}

    return one_hot_dict

def generate_endplate_one_hot(id2endplate_dict=None):

    # general one_hot_dict, for all possible endplates
    one_hot_dict = get_one_hot(get_endplate_dict()[0])

    # subset of one_hot_dict, for data found in id2endplate_dict
    if id2endplate_dict is not None:
        one_hot_dict = {id: one_hot for id, one_hot in one_hot_dict.items() if id in id2endplate_dict}

    return one_hot_dict

def get_edge_dict():

    id2edge_dict = {0: 'vert',  # between upper and lower endplate of the same vert
                    1: 'disc',  # between upper and lower endplate of different verts
                    }
    edge2id_dict = {edge_type: id for id, edge_type in id2edge_dict.items()}

    return id2edge_dict, edge2id_dict

def generate_edge_one_hot():
    one_hot_dict = get_one_hot(get_edge_dict()[0])
    return one_hot_dict

def calculate_endplate_node_features():

    # position
    # define coordinate system reference
    # use mm values

    # endplate type (upper / lower)

    # vert type

    pass

def calculate_endplate_edge_features(endplate, id, id2endplate):

    # use calculations from giraph geometric utils

    edge_id = 0 if endplate.split('_')[0] == id2endplate[id + 1].split('_')[
        0] else 1  # 0 - same vert, 1 - differet vert (disc)
    ef = [edge_one_hot[edge_id], edge_one_hot[edge_id]]

    pass

def get_disk_labels(vert_id2label):
    ids = np.array(sorted(list(vert_id2label.keys())))

    disk_id2label = {}
    for n in range(1, len(ids)):
        if ids[n] - ids[n - 1] == 1:  # adjacent verts
            id = '{}_{}'.format(ids[n], ids[n - 1])
            label = '{}_{}'.format(vert_id2label[ids[n]], vert_id2label[ids[n - 1]])
            disk_id2label[id] = label

    disk_label2id = {label: id for id, label in disk_id2label.items()}

    return disk_id2label, disk_label2id

def calc_angle_between_verts(vert1, vert2, angle_type=('lower', 'upper'), units='deg'):
    """
    Calculate angle between 2 vertebrae.

    vert1, vert2 : ndarrays
        Vertebrae corners coordinates.
        Each vert in given in the following format:

            vert = nparray(['upperVertSt_x', 'upperVertSt_y',
                           'upperVertEnd_x', 'upperVertEnd_y',
                           'lowerVertEnd_x', 'lowerVertEnd_y',
                           'lowerVertSt_x', 'lowerVertSt_y',
                           ])

    angle_type : tuple, optional
        Tuple of strings, specifies which vertebra's end plate to use.
        The first value corresponds to vert1, and the second to vert2.
        Default value is ('lower', 'upper') - appropriate for Disc angle calculation.


    """

    assert (angle_type[0] in ['upper', 'lower']) and ((angle_type[1] in ['upper', 'lower']))
    assert units[0] in ['rad', 'deg']

    v1_upper = vert1[1, :] - vert1[0, :]
    v1_lower = vert1[3, :] - vert1[2, :]
    v2_upper = vert2[1, :] - vert2[0, :]
    v2_lower = vert2[3, :] - vert2[2, :]

    v1 = v1_upper if angle_type[0] == 'upper' else v1_lower
    v2 = v2_upper if angle_type[1] == 'upper' else v2_lower

    angle = signed_angle(v1, v2, units=units)

    return angle


def signed_angle(v1, v2, look=np.array([0, 1]), units="deg"):
    """
    Adapted from: https://github.com/lace/vg/blob/44d3fd6b9e7dff0107d117d0a1beac2ec9c7ea2f/vg/core.py#L268C1-L296C51

    Compute the signed angle between two vectors.

    Results are in the range -180 and 180 (or `-math.pi` and `math.pi`). A
    positive number indicates a clockwise sweep from `v1` to `v2`. A negative
    number is counterclockwise.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as `v1`.
        look (np.arraylike): A `(3,)` vector specifying the normal of the viewing plane.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Returns:
        object: For a `(3,)` input, a `float` with the angle. For a `kx3`
        input, a `(k,)` array.
    """

    # calculate (acute) angle between vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    # The sign of (A x B) dot look gives the sign of the angle.
    # > 0 means clockwise, < 0 is counterclockwise.
    sign = np.array(np.sign(np.cross(v1, v2).dot(look)))

    # 0 means collinear: 0 or 180. Let's call that clockwise.
    sign[sign == 0] = 1

    return sign * angle(v1, v2, look, units=units)

if __name__ == '__main__':

    generate_simple_spine_graph()

    pass





