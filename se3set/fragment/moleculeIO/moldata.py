from se3set.fragment.fragmentor.base import BOFragmentor
import numpy as np
from typing import Tuple, List

def get_hypergraph_data_from_fragmentor(fragmentor: BOFragmentor, **fragment_settings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[List]]]:
    fragmentor.fragment_index(conformer_idx=0, **fragment_settings)
    hyperedge = [fragmentor.fragment_index(conformer_idx=_i, initialize=False, **fragment_settings) for _i in range(fragmentor.molecule.n_conformers)]
    atom_type = np.array([atom.atomic_number for atom in fragmentor.molecule.atoms])
    atom_formal_charge = np.array([atom.formal_charge.magnitude for atom in fragmentor.molecule.atoms])
    atom_coordinates = np.array([conformer.magnitude for conformer in fragmentor.molecule.conformers])
    return atom_type, atom_formal_charge, atom_coordinates, hyperedge

def get_single_hypergraph_data_from_fragmentor(fragmentor: BOFragmentor, conformer_idx: int = 0, **fragment_settings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List]]:
    hyperedge = fragmentor.fragment_index(conformer_idx=conformer_idx, **fragment_settings)
    atom_type = np.array([atom.atomic_number for atom in fragmentor.molecule.atoms])
    atom_formal_charge = np.array([atom.formal_charge.magnitude for atom in fragmentor.molecule.atoms])
    atom_coordinates = fragmentor.molecule.conformers[conformer_idx].magnitude
    return atom_type, atom_formal_charge, atom_coordinates, hyperedge