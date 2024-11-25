from ..moleculeIO.parse import parse_filepath, convert_rdkit_mol_to_openff_mol
from ..calculators.reference import ELEMENT_RADIUS
from ..fragmentor.fraginfo import FragGInfo

import numpy as np
from numpy import ndarray
from openff.toolkit.topology import Atom as openff_Atom
from openff.toolkit import Molecule as openff_Molecule
from openff.units import unit as ffunit
from ase import Atoms as ase_Atoms
from ase import Atom as ase_Atom

import networkx
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
from itertools import combinations
from copy import deepcopy
from abc import abstractmethod
from typing import Any, Optional, overload


class BaseFragmentor(object):
    '''
        Basic fragment Object.
    '''
    
    def __init__(self) -> None:
        pass
    
    def fragment_index(self):
        '''
        Return a fragment list with index
        '''
        pass

    @abstractmethod
    def fragment(self):
        '''
            The basic fragment method. Fragment the protein into specific sub-structures.
            Please rewrite this function in subclass according to your specific fragment method.
        '''
        pass


class BOFragmentor(BaseFragmentor):
    """Upon formation of a chemical bond, 
    the atomic electron distributions do not change very much, 
    in particular in the core regions.
    """
    def __init__(self, topoBO_threshold: float):
        self.topoBO_threshold = topoBO_threshold
        self.molecule = None
        self.BO_mat = None
        self.frags_topoBO_mat: np.ndarray
        self.groups: List[List]
        self.sorted_groups: List[List]
        self.frags: List[List]
        self.frags_init_ginfo : List[FragGInfo]
        self.frags_ginfo: List[FragGInfo]
        self.conformer_idx: int

    def load_molecule(self, input_molecule: Any, input_molecule_name: str, smiles: Optional[str] = None, mode: str = 'file', **kwargs):
        if mode == 'file':
            self.molecule = parse_filepath(input_molecule, smiles=smiles)
            self.molecule.name = input_molecule_name
            self.molecule.properties['atom_map'] = {i: i + 1 for i in range(self.molecule.n_atoms)}
        elif mode == 'openff_Molecule' and isinstance(input_molecule, openff_Molecule):
            self.molecule = input_molecule
            self.molecule.name = input_molecule_name
            self.molecule.properties['atom_map'] = {i: i + 1 for i in range(self.molecule.n_atoms)}
        elif mode == 'rdkit_Molecule':
            self.molecule = convert_rdkit_mol_to_openff_mol(input_molecule, pos=None)
        else:
            # TODO: Object input
            raise NotImplementedError("{} Not implemented.".format(mode))
            pass

    @abstractmethod
    def get_bo_mat_from_molecule(self, **kwargs) -> np.ndarray:
        '''
            Calculate different bond order matrix here
        '''
        return 
    
    @abstractmethod
    def get_frags_from_groups_by_BO(self) -> List[List]:
        '''
            Process groups to fragments according to bond order matrix here
        '''
        return

    def fragment_index(self, input_molecule: Optional[Any], input_molecule_name: Optional[str], mode: Optional[str] = 'file', smiles: Optional[str] = None, initialize: bool = True, conformer_idx: int = 0, **kwargs) -> List[List]:
        '''
            **kwargs should provide the arguments to get bo_matrix
        '''
        self.conformer_idx = conformer_idx
        if input_molecule is not None:
            self.load_molecule(input_molecule, input_molecule_name, smiles=smiles, mode=mode)
        if initialize:
            self.groups = self._fragment_from_TopoBO()
            # pre-calculated params for BO Fragmentor
            self.sorted_groups = sorted(self.groups, key=lambda x: len(x), reverse=True)
            self.frags_topoBO_mat = BOFragmentor.get_groups_TopoBO_mat(BOFragmentor.get_TopoBO_mat(self.molecule), self.sorted_groups)
        self.BO_mat = self.get_bo_mat_from_molecule(**kwargs)
        self.frags = self.get_frags_from_groups_by_BO()
        return self.frags

    def fragment(self, input_molecule: Optional[Any], input_molecule_name: Optional[str], mode: Optional[str] = 'file', smiles: Optional[str] = None, initialize: bool = True, conformer_idx: int = 0, **kwargs) -> Tuple[List[ase_Atoms], List[List]]:
        '''
            **kwargs should provide the arguments to get bo_matrix
        '''
        self.fragment_index(input_molecule, input_molecule_name, mode=mode, initialize=initialize, conformer_idx=conformer_idx, **kwargs)
        self.frag_atoms = [ase_Atoms() for _ in self.frags]
        # Append basic atoms
        self._get_ase_atoms()
        # Add Hydrogens
        self.addHs()
            
        return self.frag_atoms, self.frags

    def _fragment_from_TopoBO(self) -> List[List]:
        # prepare founded functional groups and rings
        self._prepare_molecule(self.default_functionalgroups, False)

        # get mask from funtional groups, rings and topology bond order threshold
        functional_group_mask, ring_mask = [], []
        for atom_set in self.atom_to_merged_fg.values():
            functional_group_mask += list(combinations(atom_set, 2))
        for atom_set in self.atom_to_merged_rings.values():
            ring_mask += list(combinations(atom_set, 2))
        functional_group_mask = np.array(functional_group_mask)
        ring_mask = np.array(ring_mask)
        BO_mask = BOFragmentor.get_openffMolecule_TopoBO_mask(self.molecule, threshold=self.topoBO_threshold)

        # stack all masks
        mask = np.zeros((self.molecule.n_atoms, self.molecule.n_atoms), dtype=bool)
        if functional_group_mask.shape[0] != 0 or ring_mask.shape[0] != 0 or BO_mask.shape[0] != 0:
            idx = np.unique(np.vstack([part_mask for part_mask in [functional_group_mask, ring_mask, BO_mask] if part_mask.shape[0] > 0]), axis=0).T
            mask[idx[0], idx[1]] = True
            mask[idx[1], idx[0]] = True

        # break bonds to find fragments
        adj_mat = np.zeros_like(mask, dtype=int)
        adj_mat[mask] = 1
        frags = BOFragmentor.BFS(adj_mat)

        return frags

    @staticmethod
    def get_TopoBO_mat(molecule: openff_Molecule):
        ret_mat = np.zeros((molecule.n_atoms, molecule.n_atoms), dtype=int)
        indices1 = np.array([bond.atom1_index for bond in molecule.bonds]) 
        indices2 = np.array([bond.atom2_index for bond in molecule.bonds]) 
        ret_mat[indices1, indices2] = 1
        ret_mat[indices2, indices1] = 1
        return ret_mat

    @staticmethod
    def get_groups_TopoBO_mat(atom_topo_bo_mat: np.ndarray, groups: List[List[int]]):
        groups_topo_bo_mat = np.diag(np.zeros(len(groups), dtype=float))
        for i in range(len(groups)-1):
            for j in range(i+1, len(groups)):
                groups_topo_bo_mat[i][j] = np.max(atom_topo_bo_mat[groups[i]][:, groups[j]])
        groups_topo_bo_mat = groups_topo_bo_mat + (np.triu(groups_topo_bo_mat, k=1).T)
        return groups_topo_bo_mat
    
    @staticmethod
    def get_openffMolecule_TopoBO_mask(molecule: openff_Molecule, threshold: float, disableXH: bool = True) -> np.ndarray:
        if disableXH:
            return np.array([[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds 
                            if bond.bond_order > threshold or bond.atom1.atomic_number == 1 or bond.atom2.atomic_number == 1], dtype=int)
        else:
            return np.array([[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds 
                            if bond.bond_order > threshold], dtype=int)
        
    @staticmethod
    def BFS(adj_mat: ndarray):
        num_nodes = adj_mat.shape[0]

        visited = [False] * num_nodes
        groups = []

        while False in visited:
            q = [visited.index(False)]
            tmp = [visited.index(False)]
            visited[q[0]] = True

            while q:
                vis = q.pop(0)

                for i in range(num_nodes):
                    if adj_mat[vis][i] > 0 and not visited[i]:
                        q.append(i)
                        tmp.append(i)
                        visited[i] = True

            groups.append(tmp)

        return groups

    def _postprocess_isolated_atom(self, frags: List[List[int]]):
        isolate_atom_list = [frag[0] for frag in frags if len(frag) == 1]
        new_frags = [frag for frag in frags if len(frag) > 1]

        while len(isolate_atom_list) > 0:
            atom = isolate_atom_list.pop(0)
            tmp = [atom]
            atom = np.argmax(self.BO_mat[atom])
            while atom in isolate_atom_list:
                tmp.append(atom)
                isolate_atom_list.remove(atom)
                atom = np.argmax(self.BO_mat[atom])
                if atom in tmp:
                    new_frags.append(tmp)
                    break
            else:
                for i in range(len(new_frags)):
                    if atom in new_frags[i]:
                        new_frags[i] += tmp

        return new_frags

    def _frag_index_for_atom(self, frags: List[List[int]]):
        frag_index_dict = {}
        for frag_idx, frag in enumerate(frags):
            for atom_idx in frag:
                frag_index_dict[atom_idx] = frag_idx
        return frag_index_dict
    
    @property
    def frag_index_of_atoms(self):
        frag_index_dict = {}
        for frag_idx, frag in enumerate(self.frags):
            for atom_idx in frag:
                frag_index_dict[atom_idx] = frag_idx
        return frag_index_dict

    @property
    def default_functionalgroups(self):
        """Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/data/default-functional-groups.json.
        """
        return {
            'hydrazine': '[NX3:1][NX3:2]',
            'hydrazone': '[NX3:1][NX2:2]',
            'nitric_oxide': '[N:1]-[O:2]',
            'amide': '[#7:1][#6:2](=[#8:3])',
            'amide_n': '[#7:1][#6:2](-[O-:3])',
            'amide_2': '[NX3:1][CX3:2](=[OX1:3])[NX3:4]',
            'aldehyde': '[CX3H1:1](=[O:2])[#6:3]',
            'sulfoxide_1': '[#16X3:1]=[OX1:2]',
            'sulfoxide_2': '[#16X3+:1][OX1-:2]',
            'sulfonyl': '[#16X4:1](=[OX1:2])=[OX1:3]',
            'sulfinic_acid': '[#16X3:1](=[OX1:2])[OX2H,OX1H0-:3]',
            'sulfinamide': '[#16X4:1](=[OX1:2])(=[OX1:3])([NX3R0:4])',
            'sulfonic_acid': '[#16X4:1](=[OX1:2])(=[OX1:3])[OX2H,OX1H0-:4]',
            #'selenic_acid': '[#34X4:1](=[OX1:2])(=[OX1:3])[OX2H,OX1H0-:4]',
            #'seleninic_acid': '[#34X3:1](=[OX1:2])[OX2H,OX1H0-:3]',
            #'selenamide': '[#34X4:1](=[OX1:2])(=[OX1:3])([NX3R0:4])',
            #'selenonic_acid': '[#34X4:1](=[OX1:2])(=[OX1:3])[OX2H,OX1H0-:4]',
            'phosphine_oxide': '[PX4:1](=[OX1:2])([#6:3])([#6:4])([#6:5])',
            'phosphonate': '[P:1](=[OX1:2])([OX2H,OX1-:3])([OX2H,OX1-:4])',
            'phosphate': '[PX4:1](=[OX1:2])([#8:3])([#8:4])([#8:5])',
            #'arsenic_oxide': '[AsX3:1](=[OX1:2])([#6:3])([#6:4])([#6:5])',
            #'arsenate': '[AsX4:1](=[OX1:2])([#8:3])([#8:4])([#8:5])',
            'carboxylic_acid': '[CX3:1](=[O:2])[OX1H0-,OX2H1:3]',
            'nitro_1': '[NX3+:1](=[O:2])[O-:3]',
            'nitro_2': '[NX3:1](=[O:2])=[O:3]',
            'ester': '[CX3:1](=[O:2])[OX2H0:3]',
            'tri_halide': '[#6:1]([F,Cl,I,Br:2])([F,Cl,I,Br:3])([F,Cl,I,Br:4])',
            'hydroxyl': '[#8:1]-[#1:2]',
            #'alkene': '[#6:1]=[#6:2]',
            #'alkyne': '[#6:1]#[#6:2]',
            #'high_order_bond': '[*:1]=,#[*:2]',
            'water': '[#1:1]-[#8:2]-[#1:3]'
        }

    def _get_map_index(self, atom_index: int, error_on_missing: bool=True):
        """Returns the map index of a particular atom in a molecule.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/utils.py#L41-L66.
        """
        atom_map = self.molecule.properties.get("atom_map", {})
        atom_map_index = atom_map.get(atom_index, None)

        if atom_map_index is None and error_on_missing:
            raise KeyError(f"{atom_index} is not in the atom map ({atom_map}).")

        return 0 if atom_map_index is None else atom_map_index

    def _prepare_molecule(self, functional_groups: Dict[str, str], keep_non_rotor_ring_substituents: bool):
        """Prepare a molecule for fragmentation.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/fragment.py#L801-L844.
        """
        # Find the functional groups and ring systems which should not be fragmented.
        try:
            self.functional_groups = self._find_functional_groups(functional_groups)
        except:
            # Do not use functional groups
            self.functional_groups = {}
        
        try:
            self.rings = self._find_ring_systems(self.functional_groups, keep_non_rotor_ring_substituents)
        except:
            # Do not use ring systems
            self.rings = {}

        # unique and merge adjacent functional groups
        self.atom_to_merged_fg = defaultdict(set)
        self.atom_to_merged_rings = defaultdict(set)
        
        # process functional groups
        for (atom_set, _) in self.functional_groups.values():
            for atom_idx in atom_set:
                if atom_idx in self.atom_to_merged_fg:
                    self.atom_to_merged_fg[atom_idx].update(atom_set)
                    self.atom_to_merged_fg.update({_i: self.atom_to_merged_fg[atom_idx] for _i in atom_set if _i not in self.atom_to_merged_fg})
                    break
            else:
                self.atom_to_merged_fg.update({_i: deepcopy(atom_set) for _i in atom_set})
        tmp_fgs = deepcopy(self.atom_to_merged_fg)
        for atom_idx in tmp_fgs:
            cur_atom = self.molecule.atom(atom_idx)
            for bond in cur_atom.bonds:
                adj_idx = bond.atom1_index if atom_idx == bond.atom2_index else bond.atom1_index
                if adj_idx in self.atom_to_merged_fg and self.atom_to_merged_fg[adj_idx] != self.atom_to_merged_fg[atom_idx]:
                    new_fg_atom_set = self.atom_to_merged_fg[atom_idx] | self.atom_to_merged_fg[adj_idx]
                    for idx in new_fg_atom_set:
                        self.atom_to_merged_fg[idx] = new_fg_atom_set
        # process rings
        for (atom_set, _) in self.rings.values():
            for atom_idx in atom_set:
                if atom_idx in self.atom_to_merged_rings:
                    self.atom_to_merged_rings[atom_idx].update(atom_set)
                    self.atom_to_merged_rings.update({_i: self.atom_to_merged_rings[atom_idx] for _i in atom_set if _i not in self.atom_to_merged_rings})
                    break
            else:
                self.atom_to_merged_rings.update({_i: deepcopy(atom_set) for _i in atom_set})
        '''tmp_fgs = deepcopy(self.atom_to_merged_rings)
        for atom_idx in tmp_fgs:
            cur_atom = self.molecule.atom(atom_idx)
            for bond in cur_atom.bonds:
                adj_idx = bond.atom1_index if atom_idx == bond.atom2_index else bond.atom1_index
                if adj_idx in self.atom_to_merged_rings and self.atom_to_merged_rings[adj_idx] != self.atom_to_merged_rings[atom_idx]:
                    new_ring_atom_set = self.atom_to_merged_fg[atom_idx] | self.atom_to_merged_fg[adj_idx]
                    for idx in new_ring_atom_set:
                        self.atom_to_merged_fg[idx] = new_ring_atom_set'''

        return self.functional_groups, self.rings

    def _find_functional_groups(self, functional_groups: Dict[str, str]):
        """Find the atoms and bonds involved in the functional groups specified by `functional_groups`.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/fragment.py#L245-L288.
        """
        found_groups = {}

        for functional_group, smarts in functional_groups.items():
            unique_matches = {
                tuple(sorted(match))
                for match in self.molecule.chemical_environment_matches(smarts)
            }

            for i, match in enumerate(unique_matches):
                atoms = set(index for index in match)
                bonds = set(
                    (
                        bond.atom1_index,
                        bond.atom2_index,
                    )
                    for bond in self.molecule.bonds
                    if bond.atom1_index in match and bond.atom2_index in match
                )

                found_groups[f"{functional_group}_{i}"] = (atoms, bonds)

        return found_groups

    @staticmethod
    def _find_ring_basis(molecule: openff_Molecule, max_bone_atom_num: int) -> Dict[int, int]:
        """This function attempts to find all ring systems (see [1] for more details) in
        a given molecule.

        The method first attempts to determine which atoms and bonds are part of rings
        by matching the `[*:1]@[*:2]` SMIRKS pattern against the molecule.

        The matched bonds are then used to construct a graph (using ``networkx``), from which
        the ring systems are identified as those sets of atoms which are 'connected'
        together (using ``connected_components``) by at least one path.

        Parameters
        ----------
        molecule:
            The molecule to search for ring systems.
        max_bone_atom_num:
            The maximum number of atoms in a ring system.

        Notes
        -----
        * Two molecular rings with only one common atom (i.e. spiro compounds) are
        considered to be part of the same ring system.

        References
        ----------
        [1] `Ring Perception <https://docs.eyesopen.com/toolkits/python/oechemtk/ring.html>`_

        Returns
        -------
            The index of which ring system each atom belongs to. Only ring atoms are
            included in the returned dictionary.
        """
        
        # Find the ring atoms
        ring_atom_index_pairs = {
            tuple(sorted(pair))
            for pair in molecule.chemical_environment_matches("[*:1]@[*:2]") 
        }

        # Construct a networkx graph from the found ring bonds.
        graph = networkx.Graph()

        for atom_index_pair in ring_atom_index_pairs:
            graph.add_edge(*atom_index_pair)

        ring_system_atoms = {}

        ring_cnt = 1
        for ring_system in networkx.cycles.minimum_cycle_basis(graph):  
            if len(ring_system) <= max_bone_atom_num:
                ring_system_atoms[ring_cnt] = {atom_index for atom_index in ring_system}
                ring_cnt += 1

        return ring_system_atoms

    def _find_ring_systems(self, 
                           functional_groups: Dict[str, Tuple[Set[int], Set[Tuple[int, int]]]], 
                           keep_non_rotor_ring_substituents: bool=False, max_bone_atom_num: int = 8):
        """This function finds all ring systems in a molecule.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/fragment.py#L456-L552.
        """
        ring_system_atoms = BOFragmentor._find_ring_basis(self.molecule, max_bone_atom_num=max_bone_atom_num)

        '''for ring_idx in ring_system_atoms:
            print(len(ring_system_atoms[ring_idx]))'''

        # Find the map indices of the bonds involved in each ring system.
        ring_system_bonds = defaultdict(set)

        for bond in self.molecule.bonds:
            for ring_index in ring_system_atoms:
                cur_ring = ring_system_atoms[ring_index]
                if bond.atom1_index in cur_ring and bond.atom2_index in cur_ring:
                    ring_system_bonds[ring_index].add(
                        (
                            bond.atom1_index,
                            bond.atom2_index,
                        )
                    )

        # Scan the neighbours of the ring system atoms for any functional groups
        # / non-rotor substituents which should be included in the ring systems.
        for ring_index in ring_system_atoms:
            # If any atoms are part of a functional group, include the other atoms in the
            # group in the ring system lists
            ring_functional_groups = {
                functional_group
                for map_index in ring_system_atoms[ring_index]
                for functional_group in functional_groups
                if map_index in functional_groups[functional_group][0]
            }

            ring_system_atoms[ring_index].update(
                map_index
                for functional_group in ring_functional_groups
                for map_index in functional_groups[functional_group][0]
            )
            ring_system_bonds[ring_index].update(
                map_tuple
                for functional_group in ring_functional_groups
                for map_tuple in functional_groups[functional_group][1]
            )

            if not keep_non_rotor_ring_substituents:
                continue

            non_rotor_atoms, non_rotor_bonds = self._find_non_rotor_ring_substituents(ring_system_atoms[ring_index])

            ring_system_atoms[ring_index].update(non_rotor_atoms)
            ring_system_bonds[ring_index].update(non_rotor_bonds)

        ring_systems = {
            ring_index: (
                ring_system_atoms[ring_index],
                ring_system_bonds[ring_index],
            )
            for ring_index in ring_system_atoms
        }
        return ring_systems

    def _find_non_rotor_ring_substituents(self, ring_system_atoms: Set[int]):
        """Find the non-rotor substituents attached to a particular ring system.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/fragment.py#L555.
        """
        rotatable_bonds = self.molecule.find_rotatable_bonds()

        def heavy_degree(atom: openff_Atom) -> int:
            return sum(1 for atom in atom.bonded_atoms if atom.atomic_number != 1)

        rotor_bonds = [
            bond
            for bond in rotatable_bonds
            if heavy_degree(bond.atom1) >= 2 and heavy_degree(bond.atom2) >= 2
        ]

        non_rotor_atoms = set()
        non_rotor_bonds = set()

        for bond in self.molecule.bonds:

            # Check if the bond is a rotor.
            if bond in rotor_bonds:
                continue

            if bond.atom1.atomic_number == 1 or bond.atom2.atomic_number == 1:
                continue

            map_index_1 = self._get_map_index(bond.atom1_index)
            map_index_2 = self._get_map_index(bond.atom2_index)

            in_system_1 = map_index_1 in ring_system_atoms
            in_system_2 = map_index_2 in ring_system_atoms

            if (in_system_1 and in_system_2) or (not in_system_1 and not in_system_2):
                continue

            non_rotor_atoms.update((map_index_1, map_index_2))
            non_rotor_bonds.add((map_index_1, map_index_2))

        return non_rotor_atoms, non_rotor_bonds

    def _get_ase_atoms(self):
        conformer = self.molecule.conformers[self.conformer_idx]
        for idx, frag in enumerate(self.frags):
            cur_total_charge = 0
            for atom_idx in frag:
                cur_pos = conformer[atom_idx].to(ffunit.angstrom)
                cur_atom = self.molecule.atom(atom_idx)
                cur_total_charge += int(cur_atom.formal_charge.magnitude)
                self.frag_atoms[idx].append(ase_Atom(symbol=cur_atom.symbol, position=cur_pos.magnitude, tag=-1))
            self.frag_atoms[idx].info['total_charge'] = cur_total_charge

    @staticmethod
    def get_H_position(acceptor_atom_type: str, acceptor_posi: np.ndarray, previous_atom_posi: np.ndarray) -> np.ndarray:
        """
        Obtain hydrogen position based on the acceptor atom type, receptor and acceptor relative position
        """
        difference_coordinate = previous_atom_posi - acceptor_posi
        factor = ELEMENT_RADIUS[acceptor_atom_type] + ELEMENT_RADIUS['H']
        factor /= np.linalg.norm(difference_coordinate)
        return acceptor_posi + factor * difference_coordinate
    
    def addHs(self):
        conformer = self.molecule.conformers[self.conformer_idx]
        for idx, frag in enumerate(self.frags):
            for atom_idx in frag:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in frag:
                        self.frag_atoms[idx].append(ase_Atom(symbol='H', tag=0,
                                                             position=self.get_H_position(cur_atom.symbol, conformer[atom_idx].magnitude, conformer[adj_idx].magnitude)))
            self.frag_atoms[idx].tags = [a.tag for a in self.frag_atoms[idx]]
