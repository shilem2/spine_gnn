"""
Course page:
https://geometricdeeplearning.com/lectures/

Original colab notebook:
https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR

My copy:
https://colab.research.google.com/drive/1UW-rfX-IKa4TCXF-vhjSNJmc-E-nZ-pw#scrollTo=2xcV8Yb148Kq
"""

#@title [RUN] Import python modules

import torch

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import QED, Crippen, rdMolDescriptors, rdmolops, Draw

import py3Dmol
from rdkit.Chem import AllChem

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))


from gnn.gdl_notebooks.utils import seed


#@title [RUN] Helper functions for data preparation

class SetTarget(object):
    """
    This transform mofifies the labels vector per data sample to only keep
    the label for a specific target (there are 19 targets in QM9).

    Note: for this practical, we have hardcoded the target to be target #0,
    i.e. the electric dipole moment of a drug-like molecule.
    (https://en.wikipedia.org/wiki/Electric_dipole_moment)
    """
    def __call__(self, data, target=0):
        data.y = data.y[:, target]
        return data


class CompleteGraph(object):
    """
    This transform adds all pairwise edges into the edge index per data sample,
    then removes self loops, i.e. it builds a fully connected or complete graph
    """
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


#@title [RUN] Helper functions for visualization

allowable_atoms = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "C",
    "Cl",
    "Br",
    "I",
    "H",
    "Unknown",
]

def to_atom(t):
    try:
        return allowable_atoms[int(t.argmax())]
    except:
        return "C"


def to_bond_index(t):
    t_s = t.squeeze()
    return [1, 2, 3, 4][
        int(
            torch.dot(
                t_s,
                torch.tensor(
                    range(t_s.size()[0]), dtype=torch.float, device=t.device
                ),
            ).item()
        )
    ]

def to_rdkit(data, device=None):
    has_pos = False
    node_list = []
    for i in range(data.x.size()[0]):
        node_list.append(to_atom(data.x[i][:5]))

    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    invalid_idx = set([])
    for i in range(len(node_list)):
        if node_list[i] == "Stop" or node_list[i] == "H":
            invalid_idx.add(i)
            continue
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    added_bonds = set([])
    for i in range(0, data.edge_index.size()[1]):
        ix = data.edge_index[0][i].item()
        iy = data.edge_index[1][i].item()
        bond = to_bond_index(data.edge_attr[i])  # <font color='red'>TODO</font> fix this
        # bond = 1
        # add bonds between adjacent atoms

        if data.edge_attr[i].sum() == 0:
          continue

        if (
            (str((ix, iy)) in added_bonds)
            or (str((iy, ix)) in added_bonds)
            or (iy in invalid_idx or ix in invalid_idx)
        ):
            continue
        # add relevant bond type (there are many more of these)

        if bond == 0:
            continue
        elif bond == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 2:
            bond_type = Chem.rdchem.BondType.DOUBLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 3:
            bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 4:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        added_bonds.add(str((ix, iy)))

    if has_pos:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(data.pos.size(0)):
            if i in invalid_idx:
                continue
            p = Point3D(
                data.pos[i][0].item(),
                data.pos[i][1].item(),
                data.pos[i][2].item(),
            )
            conf.SetAtomPosition(node_to_idx[i], p)
        conf.SetId(0)
        mol.AddConformer(conf)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    mol_frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    return largest_mol


def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

def smi2conf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        return None

def main():

    # Part 0: Introduction to Molecular Property Prediction with PyTorch Geometric

    # set different generators random seed
    seed(0)

    # load QM9 dataset
    path = './qm9'
    target = 0

    # Transforms which are applied during data loading:
    # (1) Fully connect the graphs, (2) Select the target/label
    transform = T.Compose([CompleteGraph(), SetTarget()])

    # Load the QM9 dataset with the transforms defined
    dataset = QM9(path, transform=transform)

    # Normalize targets per data sample to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

    # The QM9 dataset has over 130,000 molecular graphs!
    # Let us create a more tractable sub-set of 3,000 molecular graphs for the purposes of this practical and separate it into training, validation, and test sets.

    print(f"Total number of samples: {len(dataset)}.")

    # Split datasets (in case of using the full dataset)
    # test_dataset = dataset[:10000]
    # val_dataset = dataset[10000:20000]
    # train_dataset = dataset[20000:]

    # Split datasets (our 3K subset)
    train_dataset = dataset[:1000]
    val_dataset = dataset[1000:2000]
    test_dataset = dataset[2000:3000]
    print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # Create dataloaders with batch size = 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Visualising Molecular Graphs
    # Note: we have implemented some code for you to convert the PyG graph into a Molecule object that can be used by RDKit, a python package for chemistry and visualing molecules. It is not important for you to understand RDKit beyond visualisation purposes.
    num_viz = 50
    mols = [to_rdkit(train_dataset[i]) for i in range(num_viz)]
    values = [str(round(float(train_dataset[i].y), 3)) for i in range(num_viz)]

    fig = Draw.MolsToGridImage(mols, legends=[f"y = {value}" for value in values], molsPerRow=5)
    fig.show()

    # Understanding PyG Data Objects
    # Each graph in our dataset is encapsulated in a PyG `Data` object, a convient way of representing all structured data for use in Geometric Deep Learning (including graphs, point clouds, and meshes).
    data = train_dataset[0]  # one data sample, i.e. molecular graph
    print("Let us print all the attributes (along with their shapes) that our PyG molecular graph contains:")
    print(data)

    print(f"\nThis molecule has {data.x.shape[0]} atoms, and {data.edge_attr.shape[0]} edges.")
    print(f"\nFor each atom, we are given a feature vector with {data.x.shape[1]} entries (described above).")
    print(f"\nFor each edge, we are given a feature vector with {data.edge_attr.shape[1]} entries (also described above).")
    print(f"\nIn the next section, we will learn how to build a GNN in the Message Passing flavor to process the node and edge features of molecular graphs and predict their properties.")
    print(f"\nEach atom also has a {data.pos.shape[1]}-dimensional coordinate associated with it. We will talk about their importance later in the practical.")
    print(f"\nFinally, we have {data.y.shape[0]} regression target for the entire molecule.")

    # Using PyG for batching
    # Toy graph 1
    edge_index_1 = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long
    )
    x_1 = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data_1 = Data(x=x_1, edge_index=edge_index_1)

    # Toy graph 2
    edge_index_2 = torch.tensor(
        [[0, 2, 1, 0], [2, 0, 0, 1]],
        dtype=torch.long
    )
    x_2 = torch.tensor([[1], [0], [-1]], dtype=torch.float)

    data_2 = Data(x=x_2, edge_index=edge_index_2)

    # Create batch from toy graphs
    data_list = [data_1, data_2]
    batch = Batch.from_data_list(data_list)

    assert (batch[0].x == data_1.x).all() and (batch[1].x == data_2.x).all()

    # Create DataLoader
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    it = iter(loader)
    batch_1 = next(it)
    batch_2 = next(it)

    assert (batch_1.x == data_1.x).all() and (batch_2.x == data_2.x).all()

    pass




if __name__ == '__main__':

    main()

    pass