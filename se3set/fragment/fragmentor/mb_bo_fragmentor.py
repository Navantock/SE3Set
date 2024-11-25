from ..fragmentor.mg_bo_fragmentor import MergeGroupsBOFragmentor, LooseFG_MergeGroupsBOFragmentor

from ase import Atoms as ase_Atoms
from ase import Atom as ase_Atom
from openff.units import unit as ffunit

from typing import Any, Optional
import warnings


class MonomerBOFragmentor(MergeGroupsBOFragmentor):
    def __init__(self, 
                 topoBO_threshold: float,  
                 min_kernel_atoms_num: int, 
                 min_bo_threshold: float = 0.2,
                 expand_fbo_threshold: float=0.01,
                 max_kernel_atoms_num: Optional[int] = None,
                 **kwargs):
        
        super().__init__(min_kernel_atoms_num=min_kernel_atoms_num,
                         max_kernel_atoms_num=max_kernel_atoms_num,
                         min_bo_threshold=min_bo_threshold,
                         expand_fbo_threshold=expand_fbo_threshold,
                         topoBO_threshold=topoBO_threshold,
                         **kwargs)
        self.nmer_ = 1
        self.frags = []
        self.overlap_frags = []
        self.addH_indices = {}
    
    def get_nmer(self, _n: int):
        assert _n >=1, "{}-mer is invalid.".format(_n)
        if _n == 1:
            return self.frags + self.overlap_frags
        else:
            return None

    def get_nmer_atoms(self, _n: int):
        assert _n >=1, "{}-mer is invalid.".format(_n)
        if _n == 1:
            return self.monomer_atoms
        else:
            return None
    
    def get_nmer_idx(self, _n: int):
        assert _n >=1, "{}-mer is invalid.".format(_n)
        if _n == 1:
            return ['{}-f'.format(_) for _ in range(len(self.frags))] + ['{}-o'.format(_) for _ in range(len(self.overlap_frags))]
        else:
            return None
    
    def fragment(self, input_molecule: Any, input_molecule_name: str, mode: str = 'file', smiles: Optional[str] = None, initialize: bool = True, conformer_idx: int = 0, add_hydrogens: bool = False, **kwargs):
        super(MonomerBOFragmentor).fragment_index(input_molecule, input_molecule_name, mode=mode, smiles=smiles, conformer_idx=conformer_idx, initialize=initialize, **kwargs)
        self.monomer_atoms = [ase_Atoms() for _ in self.get_nmer(1)]
        # Append basic atoms
        self._get_ase_atoms()
        # Add Hydrogens
        if add_hydrogens:
            self.addHs()
            
        return self.monomer_atoms, self.get_nmer(1)
                    
    def _get_ase_atoms(self):
        conformer = self.molecule.conformers[self.conformer_idx]
        for idx, frag in enumerate(self.get_nmer(1)):
            cur_total_charge = 0
            for atom_idx in frag:
                cur_pos = conformer[atom_idx].to(ffunit.angstrom)
                cur_atom = self.molecule.atom(atom_idx)
                cur_total_charge += int(cur_atom.formal_charge.magnitude)
                self.monomer_atoms[idx].append(ase_Atom(symbol=cur_atom.symbol, position=cur_pos.magnitude, tag=-1))
            self.monomer_atoms[idx].info['total_charge'] = cur_total_charge

    def addHs(self):
        conformer = self.molecule.conformers[self.conformer_idx]

        monomer_idx = self.get_nmer_idx(1)
        self.addH_indices.update({_: [] for _ in self.get_nmer_idx(1)})
        for idx, frag in enumerate(self.get_nmer(1)):
            for atom_idx in frag:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in frag:
                        if adj_idx in self.addH_indices[monomer_idx[idx]]:
                            warnings.warn("Monomer {} duplicated addH at atom index {}.".format(idx, adj_idx))
                        self.addH_indices[monomer_idx[idx]].append(adj_idx)
                        self.monomer_atoms[idx].append(ase_Atom(symbol='H', tag=0,
                                                             position=self.get_H_position(cur_atom.symbol, conformer[atom_idx].magnitude, conformer[adj_idx].magnitude)))
            self.monomer_atoms[idx].tags = [a.tag for a in self.monomer_atoms[idx]]


class CappedMonomerBOFragmentor(MonomerBOFragmentor):
    def __init__(self, 
                 overlap_level: str,
                 topoBO_threshold: float, 
                 min_kernel_atoms_num: int, 
                 min_bo_threshold: float = 0.2, 
                 expand_fbo_threshold: float = 0.01, 
                 max_kernel_atoms_num: Optional[int] = None,
                 expand_fbo_method: str = "sum",
                 **kwargs):
        super().__init__(topoBO_threshold, min_kernel_atoms_num, min_bo_threshold, expand_fbo_threshold, max_kernel_atoms_num, **kwargs)
        self.overlap_method = None
        if overlap_level == "groups":
            if expand_fbo_method == "sum":
                self.overlap_method = super()._get_grouplevel_overlaps_sum
            elif expand_fbo_method == "max":
                self.overlap_method = super()._get_grouplevel_overlaps_max
            else:
                raise NotImplementedError("Unsupport Expand FBO Method: {}. Current Support Expand FBO Method: sum, max")
        elif overlap_level == "frags":
            if expand_fbo_method == "sum":
                self.overlap_method = super()._get_fraglevel_overlaps_sum
            elif expand_fbo_method == "max":
                self.overlap_method = super()._get_fraglevel_overlaps_max
            else:
                raise NotImplementedError("Unsupport Expand FBO Method: {}. Current Support Expand FBO Method: sum, max")
        elif overlap_level == "topo":
            self.overlap_method = super()._get_topolevel_overlaps
        else:
            raise NotImplementedError("Unsupport Overlap Level: {}. Current Support Overlap Level: groups, frags, topo")

    def fragment_index(self, input_molecule: Any, input_molecule_name: str, mode: str = 'file', smiles: Optional[str] = None, initialize: bool = True, conformer_idx: int = 0, **kwargs):
        super().fragment_index(input_molecule, input_molecule_name, mode, smiles=smiles, initialize=initialize, conformer_idx=conformer_idx,**kwargs)
        if len(self.frags) > 1:
            self.overlap_method()
        return self.get_nmer(1)
