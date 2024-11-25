from typing import Sequence, Optional, List
import numpy as np


class FragGInfo(object):
    def __init__(self, idx: int, gindices: Sequence, natoms: int, centrality: Optional[float] = None, **kwargs) -> None:
        self.idx = idx
        self.gindices = list(gindices)
        self.n_atoms = natoms
        self.valid = True
        self.centrality = centrality

    def get_atom_indices(self, groups: List[List]):
        atom_indices = []
        for g_idx in self.gindices:
            atom_indices += groups[g_idx]
        return atom_indices

    def merge(self, another, frags_bo_mat: np.ndarray):
        self.gindices.extend(another.gindices)
        self.n_atoms += another.n_atoms
        self.centrality = np.sum(frags_bo_mat[self.idx][frags_bo_mat[self.idx] >= 0])
        another.valid = False
        
    def __eq__(self, another):
        return self.idx == another.idx

    def __lt__(self, another):
        return (self.n_atoms, self.centrality) < (another.n_atoms, another.centrality)

    def __gt__(self, another):
        return (self.n_atoms, self.centrality) > (another.n_atoms, another.centrality)