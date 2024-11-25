from ..fragmentor.base import BOFragmentor
from ..fragmentor.fraginfo import FragGInfo

import numpy as np

from typing import Dict, List
from collections import defaultdict
import copy
from copy import deepcopy


class MergeGroupsBOFragmentor(BOFragmentor):
    def __init__(self, 
                 min_kernel_atoms_num: int, 
                 max_kernel_atoms_num: int, 
                 min_bo_threshold: float, 
                 expand_fbo_threshold: float, 
                 topoBO_threshold: float):
        super().__init__(topoBO_threshold)
        self.min_kernel_atoms_num = min_kernel_atoms_num
        self.max_kernel_atoms_num = max_kernel_atoms_num
        self.min_bo_threshold = min_bo_threshold
        self.expand_fbo_threshold = expand_fbo_threshold
        self.merge_frag_method = self.merge_frags

    def get_frags_from_groups_by_BO(self) -> List[List]:
        # merge frag method use @argument::min_bo_threshold to determine whether to exclude a fragment or not
        return self.merge_frag_method()
    
    @staticmethod
    def get_frags_BO_mat(atom_bo_mat: np.ndarray, frags: List[List[int]]):
        frags_bo_mat = np.diag(np.zeros(len(frags), dtype=float))
        for i in range(len(frags)-1):
            for j in range(i+1, len(frags)):
                frags_bo_mat[i][j] = np.max(atom_bo_mat[frags[i]][:, frags[j]])
        frags_bo_mat = frags_bo_mat + (np.triu(frags_bo_mat, k=1).T)
        return frags_bo_mat

    @staticmethod
    def update_frags_BO_mat(my_idx: int, merge_idx: int, frags_bo_mat: np.ndarray):
        frags_bo_mat[my_idx] = np.max(frags_bo_mat[[my_idx, merge_idx]], axis=0)
        frags_bo_mat[:, my_idx] = np.max(frags_bo_mat[:, [my_idx, merge_idx]], axis=1)
        frags_bo_mat[my_idx][my_idx] = 0
        frags_bo_mat[merge_idx] = -1
        frags_bo_mat[:, merge_idx] = -1

    @staticmethod
    def get_frags_sum_FBO_mat(atom_bo_mat: np.ndarray, frags: List[List[int]]):
        frags_fbo_mat = np.diag(-np.ones(len(frags), dtype=float))
        for i in range(len(frags)-1):
            for j in range(i+1, len(frags)):
                frags_fbo_mat[i][j] = np.sum(atom_bo_mat[frags[i]][:, frags[j]])
        frags_fbo_mat = frags_fbo_mat + (np.triu(frags_fbo_mat, k=1).T)
        return frags_fbo_mat

    @staticmethod
    def get_frags_max_FBO_mat(atom_bo_mat: np.ndarray, frags: List[List[int]]):
        frags_fbo_mat = np.diag(-np.ones(len(frags), dtype=float))
        for i in range(len(frags)-1):
            for j in range(i+1, len(frags)):
                frags_fbo_mat[i][j] = np.max(atom_bo_mat[frags[i]][:, frags[j]])
        frags_fbo_mat = frags_fbo_mat + (np.triu(frags_fbo_mat, k=1).T)
        return frags_fbo_mat
    
    @staticmethod
    def get_frags2frags_FBO_mat(atom_bo_mat: np.ndarray, fragsA: List[List[int]], fragsB: List[List[int]]):
        frags_fbo_mat = np.zeros((len(fragsA), len(fragsB)), dtype=float)
        for i in range(len(fragsA)):
            for j in range(len(fragsB)):
                frags_fbo_mat[i][j] = np.max(atom_bo_mat[fragsA[i]][:, fragsB[j]])
        return frags_fbo_mat

    def merge_frags(self) -> List[set]:
        if self.molecule.n_atoms <= self.min_kernel_atoms_num or len(self.sorted_groups) <= 1:
            return [list(range(self.molecule.n_atoms))]

        # Fragments maximum BO Matrix
        frags_bo_mat = MergeGroupsBOFragmentor.get_frags_BO_mat(self.BO_mat, self.sorted_groups)
        # Sort the initialized frags by (# atoms, BO Summation)
        self.frags_init_ginfo = [FragGInfo(g_idx, [g_idx], natoms=len(g), centrality=np.sum(frags_bo_mat[g_idx][frags_bo_mat[g_idx] >= 0])) for g_idx, g in enumerate(self.sorted_groups)]
        self.frags_ginfo = sorted(self.frags_init_ginfo, key=lambda x: (x.n_atoms, x.centrality), reverse=True) # shadow copy, keep the reference
        
        # Merge the fragment with atoms less than @argument::min_atoms_num to another fragment
        self.isolate_frags_ginfo = []
        while self.frags_ginfo[-1].n_atoms < self.min_kernel_atoms_num:
            merge_frag_ginfo = self.frags_ginfo.pop(-1)
            merge_frag_idx = merge_frag_ginfo.idx
            # Try to find a fragment to merge

            # Fisrt try topology adjacent, try to merge by order (n_atoms, centrality)
            topo_adj_frag_idx = np.where(self.frags_topoBO_mat[merge_frag_idx] >= 1)[0]
            recv_frag_idx = None
            if len(topo_adj_frag_idx) > 0:
                recv_frag_idx = min(topo_adj_frag_idx, key=lambda x: self.frags_init_ginfo[x])
            if recv_frag_idx is None or merge_frag_ginfo.n_atoms + self.frags_init_ginfo[recv_frag_idx].n_atoms > self.max_kernel_atoms_num:
                # if no topology adjacent frags found, or max_kernel_atoms_num not satisfied, then try to find the minimum frags to merge
                bo_adj_frag_idx = np.where(frags_bo_mat[merge_frag_idx] >= self.min_bo_threshold)[0]
                if len(bo_adj_frag_idx) > 0:
                    recv_frag_idx = min(bo_adj_frag_idx, key=lambda x: self.frags_init_ginfo[x])
                else:
                    # if no fragment found within min threshold, this fragment are seperated alone
                    self.isolate_frags_ginfo.append(merge_frag_ginfo)
                    continue
            
            # Update the matrix and merge the two fragments
            MergeGroupsBOFragmentor.update_frags_BO_mat(recv_frag_idx, merge_frag_idx, frags_bo_mat)
            MergeGroupsBOFragmentor.update_frags_BO_mat(recv_frag_idx, merge_frag_idx, self.frags_topoBO_mat)
            self.frags_init_ginfo[recv_frag_idx].merge(self.frags_init_ginfo[merge_frag_idx], frags_bo_mat)
            
            # Resort the new fragment
            '''adjust_idx = self.frags_ginfo.index(recv_frag_ginfo)
            bisect.insort_left(self.frags_ginfo, self.frags_ginfo.pop(adjust_idx), key=lambda x: (x.n_atoms, x.centrality)))'''
            self.frags_ginfo.sort(key=lambda x: (x.n_atoms, x.centrality), reverse=True)

        return [fg.get_atom_indices(self.sorted_groups) for fg in self.frags_ginfo + self.isolate_frags_ginfo]

    def _extend_margin_atoms(self, _atom_idx: int, _adj_idx: int, cur_ol: set):
        if _atom_idx in cur_ol:
            return []

        # save the exist overlap atom
        if _atom_idx in self.atom_to_merged_rings:
            cur_ring = list(self.atom_to_merged_rings[_atom_idx])
            ret_ring = copy.deepcopy(cur_ring)
            for atom_idx in cur_ring:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in cur_ol and adj_idx not in cur_ring and adj_idx != _adj_idx:
                        adj_atom = self.molecule.atom(adj_idx)
                        if adj_atom.symbol == 'H':
                            ret_ring.append(adj_idx)
                        '''else:
                            ret_ring.append(adj_idx)
                            for cap_atom in adj_atom.bonded_atoms:
                                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index != atom_idx:
                                    ret_ring.append(cap_atom.molecule_atom_index)'''
            return ret_ring
        if _atom_idx in self.atom_to_merged_fg:
            cur_fg = list(self.atom_to_merged_fg[_atom_idx])
            ret_fg = copy.deepcopy(cur_fg)
            for atom_idx in cur_fg:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in cur_ol and adj_idx not in cur_fg and adj_idx != _adj_idx:
                        adj_atom = self.molecule.atom(adj_idx)
                        if adj_atom.symbol == 'H':
                            ret_fg.append(adj_idx)
                        else:
                            ret_fg.append(adj_idx)
                            for cap_atom in adj_atom.bonded_atoms:
                                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index != atom_idx:
                                    ret_fg.append(cap_atom.molecule_atom_index)
            return ret_fg

        else:
            ret = [_atom_idx]
            cur_atom = self.molecule.atom(_atom_idx)
            for cap_atom in cur_atom.bonded_atoms:
                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index not in cur_ol and cap_atom.molecule_atom_index != _adj_idx:
                    ret.append(cap_atom.molecule_atom_index)
            return ret

    def _get_grouplevel_overlaps_sum(self):
        groups_fbo_mat = MergeGroupsBOFragmentor.get_frags_sum_FBO_mat(self.BO_mat, self.sorted_groups)
        for frag_idx, frag_ginfo in enumerate(self.frags_ginfo):
            frag_to_group_fbo = np.sum(groups_fbo_mat[frag_ginfo.gindices], axis=0)
            frag_to_group_fbo[frag_ginfo.gindices] = 0
            expand_indices = np.where(frag_to_group_fbo >= self.expand_fbo_threshold)[0]
            self.frags[frag_idx] += sum([self.sorted_groups[idx] for idx in expand_indices], [])

    def _get_grouplevel_overlaps_max(self):
        groups_fbo_mat = MergeGroupsBOFragmentor.get_frags_max_FBO_mat(self.BO_mat, self.sorted_groups)
        for frag_idx, frag_ginfo in enumerate(self.frags_ginfo):
            frag_to_group_fbo = np.max(groups_fbo_mat[frag_ginfo.gindices], axis=0)
            frag_to_group_fbo[frag_ginfo.gindices] = 0
            expand_indices = np.where(frag_to_group_fbo >= self.expand_fbo_threshold)[0]
            self.frags[frag_idx] += sum([self.sorted_groups[idx] for idx in expand_indices], [])
    
    def _get_fraglevel_overlaps_sum(self):
        frags_fbo_mat = MergeGroupsBOFragmentor.get_frags_sum_FBO_mat(self.BO_mat, self.frags)
        for frag_idx in range(len(self.frags)):
            expand_indices = np.where(frags_fbo_mat[frag_idx] >= self.expand_fbo_threshold)[0]
            self.frags[frag_idx] += sum([self.frags[idx] for idx in expand_indices], [])

    def _get_fraglevel_overlaps_max(self):
        frags_fbo_mat = MergeGroupsBOFragmentor.get_frags_max_FBO_mat(self.BO_mat, self.frags)
        for frag_idx in range(len(self.frags)):
            expand_indices = np.where(frags_fbo_mat[frag_idx] >= self.expand_fbo_threshold)[0]
            self.frags[frag_idx] += sum([self.frags[idx] for idx in expand_indices], [])

    def _get_topolevel_overlaps(self):
        frag_index_dict = self._frag_index_for_atom(self.frags)
        frag_indices2atom_idx = defaultdict(set)
        new_frags = copy.deepcopy(self.frags)
        for idx, frag in enumerate(self.frags):
            for atom_idx in frag:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    adj_frag_idx = frag_index_dict[adj_idx]
                    if adj_frag_idx > idx:
                        cur_cap = self._extend_margin_atoms(adj_idx, atom_idx, frag_indices2atom_idx[(idx, adj_frag_idx)])
                        adj_cap = self._extend_margin_atoms(atom_idx, adj_idx, frag_indices2atom_idx[(idx, adj_frag_idx)])
                        new_frags[idx] += cur_cap
                        new_frags[adj_frag_idx] += adj_cap

                        new_ol = cur_cap + adj_cap
                        frag_indices2atom_idx[(idx, adj_frag_idx)] |= set(new_ol)

        self.frag_indices2atom_idx = frag_indices2atom_idx
        self.overlap_frags = [list(frag_indices2atom_idx[(idx1, idx2)]) for idx1 in range(len(self.frags) - 1) for idx2 in range(idx1+1, len(self.frags)) if (idx1, idx2) in frag_indices2atom_idx]
        self.ffol_indices = [(idx1, idx2) for idx1 in range(len(self.frags) - 1) for idx2 in range(idx1+1, len(self.frags)) if (idx1, idx2) in frag_indices2atom_idx]
        self.frags = new_frags


class LooseFG_MergeGroupsBOFragmentor(MergeGroupsBOFragmentor):
    def _prepare_molecule(self, functional_groups: Dict[str, str], keep_non_rotor_ring_substituents: bool):
        """Prepare a molecule for fragmentation.
        Refer to https://github.com/openforcefield/openff-fragmenter/blob/main/openff/fragmenter/fragment.py#L801-L844.
        """
        # Find the functional groups and ring systems which should not be fragmented.
        self.functional_groups = self._find_functional_groups(functional_groups)
        self.rings = self._find_ring_systems(self.functional_groups, keep_non_rotor_ring_substituents)

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
        # Do not merge adjacent functional groups

        # process rings
        for (atom_set, _) in self.rings.values():
            for atom_idx in atom_set:
                if atom_idx in self.atom_to_merged_rings:
                    self.atom_to_merged_rings[atom_idx].update(atom_set)
                    self.atom_to_merged_rings.update({_i: self.atom_to_merged_rings[atom_idx] for _i in atom_set if _i not in self.atom_to_merged_rings})
                    break
            else:
                self.atom_to_merged_rings.update({_i: deepcopy(atom_set) for _i in atom_set})

        return self.functional_groups, self.rings

    def _extend_margin_atoms(self, _atom_idx: int, _adj_idx: int, cur_ol: set):
        if _atom_idx in cur_ol:
            return []

        # save the exist overlap atom
        if _atom_idx in self.atom_to_merged_rings:
            cur_ring = list(self.atom_to_merged_rings[_atom_idx])
            ret_ring = copy.deepcopy(cur_ring)
            for atom_idx in cur_ring:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in cur_ol and adj_idx not in cur_ring and adj_idx != _adj_idx:
                        adj_atom = self.molecule.atom(adj_idx)
                        if adj_atom.symbol == 'H':
                            ret_ring.append(adj_idx)
                        '''else:
                            ret_ring.append(adj_idx)
                            for cap_atom in adj_atom.bonded_atoms:
                                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index != atom_idx:
                                    ret_ring.append(cap_atom.molecule_atom_index)'''
            return ret_ring
        if _atom_idx in self.atom_to_merged_fg:
            cur_fg = list(self.atom_to_merged_fg[_atom_idx])
            ret_fg = copy.deepcopy(cur_fg)
            for atom_idx in cur_fg:
                cur_atom = self.molecule.atom(atom_idx)
                for bond in cur_atom.bonds:
                    adj_idx = bond.atom2_index if atom_idx == bond.atom1_index else bond.atom1_index
                    if adj_idx not in cur_ol and adj_idx not in cur_fg and adj_idx != _adj_idx:
                        adj_atom = self.molecule.atom(adj_idx)
                        if adj_atom.symbol == 'H':
                            ret_fg.append(adj_idx)
                        '''else:
                            ret_fg.append(adj_idx)
                            for cap_atom in adj_atom.bonded_atoms:
                                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index != atom_idx:
                                    ret_fg.append(cap_atom.molecule_atom_index)'''
            return ret_fg

        else:
            ret = [_atom_idx]
            cur_atom = self.molecule.atom(_atom_idx)
            for cap_atom in cur_atom.bonded_atoms:
                if cap_atom.symbol == 'H' and cap_atom.molecule_atom_index not in cur_ol and cap_atom.molecule_atom_index != _adj_idx:
                    ret.append(cap_atom.molecule_atom_index)
            return ret