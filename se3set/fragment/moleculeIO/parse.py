import numpy as np
import rdkit.Chem as rdChem
from openff.toolkit import Molecule
from openff.units import unit as ffunit
from openff.units import Quantity
from typing import Optional, Sequence
import warnings


def convert_rdkit_mol_to_openff_mol(mol, pos):
    try:
        openff_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        openff_mol.add_conformer(Quantity(np.array(pos), ffunit.angstrom))
        return openff_mol
    except:
        openff_mol = Molecule()
        for atom in mol.GetAtoms():
            openff_mol.add_atom(atom.GetAtomicNum(), formal_charge=atom.GetFormalCharge(), is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            openff_mol.add_bond(a1, a2, bond_order=bond.GetBondType(), is_aromatic=bond.GetIsAromatic())
        openff_mol.add_conformer(Quantity(np.array(pos), ffunit.angstrom))
        return openff_mol

def parse_filepath(fpath: str, is_polymer: bool = True, smiles: Optional[str] = None) -> Molecule:
    '''
    Parse a structure file to OpenFF Molecule
    '''
    suffix_name = fpath.split('.')[-1]
    if suffix_name == 'pdb':
        if smiles is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                molecule = Molecule.from_pdb_and_smiles(fpath, smiles, allow_undefined_stereo=True)
            return molecule
        elif is_polymer:
            return Molecule.from_polymer_pdb(fpath)
        else:
            mol = rdChem.MolFromPDBFile(fpath, sanitize=True, removeHs=False)
            assert mol is not None, "Failed to parse PDB file {}".format(fpath)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
            #print(molecule.n_atoms, molecule.total_charge)
            return molecule
    elif suffix_name == 'sdf':
        return Molecule.from_file(fpath, allow_undefined_stereo=True)
    raise TypeError('{} : Unsupported File Type'.format(fpath))

def parse_mol_data_with_single_conformer(conformer: Sequence[Sequence[float]], smiles: str, explicit_H: bool = True, coodinate_unit = ffunit.angstrom) -> Molecule:
    '''
    Parse a structure file to OpenFF Molecule
    '''
    rdkit_mol = rdChem.MolFromSmiles(smiles, sanitize=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        molecule = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=explicit_H)
        molecule.add_conformer(Quantity(np.array(conformer), ffunit.angstrom))
    return molecule