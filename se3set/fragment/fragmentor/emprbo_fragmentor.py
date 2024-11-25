from ..fragmentor.mg_bo_fragmentor import MergeGroupsBOFragmentor
from ..fragmentor.mb_bo_fragmentor import MonomerBOFragmentor, CappedMonomerBOFragmentor
from ..calculators.bo_calculator import (get_Lendvay_empirical_bomat_from_molecule, 
                                         get_SimpleExp_bomat_from_molecule)

from typing import Any, Optional


class MG_EmprBOFragmentor(MergeGroupsBOFragmentor):
    def __init__(self, 
                 topoBO_threshold: float,  
                 min_kernel_atoms_num: int, 
                 min_bo_threshold: float = 0.2,
                 expand_fbo_threshold: float = 0.01,
                 max_kernel_atoms_num: Optional[int] = None,
                 empr_bo_method: str = 'SimpleExp',
                 **kwargs):

        super().__init__(min_kernel_atoms_num=min_kernel_atoms_num,
                         max_kernel_atoms_num=max_kernel_atoms_num,
                         min_bo_threshold=min_bo_threshold,
                         expand_fbo_threshold=expand_fbo_threshold,
                         topoBO_threshold=topoBO_threshold,
                         **kwargs)

        if empr_bo_method == 'Lendvay':
            self.emprbo_calculator = get_Lendvay_empirical_bomat_from_molecule
        elif empr_bo_method == 'SimpleExp':
            self.emprbo_calculator = get_SimpleExp_bomat_from_molecule
        else:
            raise NotImplementedError(f'Empirical Bond Order Calculator {empr_bo_method} is not implemented yet.')

    def get_bo_mat_from_molecule(self, empirical_a_param: float = 0.25, **kwargs):
        return self.emprbo_calculator(self.molecule, _a = empirical_a_param, conformer_idx=self.conformer_idx)

    def fragment_index(self, input_molecule: Any, input_molecule_name: str, mode: str = 'file', smiles: Optional[str] = None, conformer_idx: int = 0, initialize: bool = True, mult: int = 1):
        return super().fragment_index(input_molecule, input_molecule_name, mode, smiles=smiles, initialize=initialize, conformer_idx=conformer_idx, mult=mult)

    def fragment(self, input_molecule: Any, input_molecule_name: str, mode: str = 'file', smiles: Optional[str] = None, initialize: bool = True, conformer_idx: int = 0, mult: int = 1):
        return super().fragment(input_molecule, input_molecule_name, mode, smiles=smiles, initialize=initialize, conformer_idx=conformer_idx, mult=mult)


class Monomer_EmprBOFragmentor(MG_EmprBOFragmentor, MonomerBOFragmentor):
    def __init__(self, 
                 topoBO_threshold: float,  
                 min_kernel_atoms_num: int, 
                 min_bo_threshold: float = 0.2,
                 max_kernel_atoms_num: Optional[int] = None,
                 **kwargs):
        
        super().__init__(min_kernel_atoms_num=min_kernel_atoms_num,
                         max_kernel_atoms_num=max_kernel_atoms_num,
                         min_bo_threshold=min_bo_threshold,
                         topoBO_threshold=topoBO_threshold,
                         **kwargs)


class CappedMonomer_EmprBOFragmentor(MG_EmprBOFragmentor, CappedMonomerBOFragmentor):
    def __init__(self, 
                 overlap_level: str,
                 topoBO_threshold: float,  
                 min_kernel_atoms_num: int, 
                 min_bo_threshold: float = 0.2,
                 expand_fbo_threshold: float=0.01,
                 max_kernel_atoms_num: Optional[int] = None,
                 expand_fbo_method: str = 'sum',
                 **kwargs):
        
        super().__init__(min_kernel_atoms_num=min_kernel_atoms_num,
                         max_kernel_atoms_num=max_kernel_atoms_num,
                         min_bo_threshold=min_bo_threshold,
                         expand_fbo_threshold=expand_fbo_threshold,
                         topoBO_threshold=topoBO_threshold,
                         overlap_level=overlap_level,
                         expand_fbo_method=expand_fbo_method,
                         **kwargs)