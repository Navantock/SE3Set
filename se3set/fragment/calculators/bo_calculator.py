import numpy as np
from openff.toolkit import Molecule
from openff.units import unit as ffunit
from ..calculators.reference import ELEMENT_RADIUS


def _calc_distance_matrix(_positions: np.ndarray) -> np.ndarray:
    _G = np.dot(_positions, _positions.T)
    _H = np.tile(np.diag(_G), (_positions.shape[0],1))
    return np.sqrt(_H + _H.T - 2.*_G)

def get_Lendvay_empirical_bomat_from_molecule(molecule: Molecule, _a: float = 0.25, conformer_idx: int = 0, **kwargs) -> np.ndarray:
    '''
        The Bond Order refers to Lendvay, György. “On the correlation of bond order and bond length.” Journal of Molecular Structure-theochem 501 (2000): 389-393. https://doi.org/10.1016/S0166-1280(99)00449-2
    '''
    dis = _calc_distance_matrix(((molecule.conformers[conformer_idx]).to(ffunit.angstrom)).magnitude)

    r_l = np.array([[ELEMENT_RADIUS[atom.symbol] for atom in molecule.atoms]], dtype=float)
    r_l = np.tile(r_l, (r_l.shape[-1], 1))
    r_e = r_l + r_l.T
    return np.exp(-(dis -r_e) * r_e / 0.25)

def get_SimpleExp_bomat_from_molecule(molecule: Molecule, conformer_idx: int = 0, **kwargs) -> np.ndarray:
    '''
        The Bond Order refers to Pauling, Linus. “The nature of the chemical bond. IV. The energy of single bonds and the relative electronegativity of atoms.” Journal of the American Chemical Society 54.9 (1932): 3570-3582. https://doi.org/10.1021/ja01348a011
    '''
    dis = _calc_distance_matrix(((molecule.conformers[conformer_idx]).to(ffunit.angstrom)).magnitude)

    r_l = np.array([[ELEMENT_RADIUS[atom.symbol] for atom in molecule.atoms]], dtype=float)
    r_l = np.tile(r_l, (r_l.shape[-1], 1))
    r_e = r_l + r_l.T
    return np.exp(-dis + r_e)