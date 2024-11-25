from .qm9 import Hypergraph_QM9
from .hg_data import build_data

from torch_geometric.datasets import QM9

import torch
import numpy as np

import os
import shutil
from typing import Optional, Callable
from tqdm import tqdm


class Hypergraph_QM9_BRICS_Hijack(QM9):
    def __init__(self, 
                 root: str, 
                 copy_dataset: Hypergraph_QM9, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        # Attached Fragmentor Arguments
        self.processed_info_dir = os.path.join(root, "processed_info")
        if not os.path.exists(self.processed_info_dir):
            os.makedirs(self.processed_info_dir)
        shutil.copy(os.path.join(copy_dataset.processed_info_dir, "invalid.npz"), self.processed_info_dir)
        self.copy_dataset = copy_dataset

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem import BRICS
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        sanitized_suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)
        invalid_indices = np.load(os.path.join(self.processed_info_dir, "invalid.npz"))["invalid_idx"]

        data_list = []
        cnt = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i in invalid_indices:
                continue
            
            # Get data from copy_dataset
            data = self.copy_dataset[cnt]
            cnt += 1
            # BRICS Fragmentation
            # Refer to https://github.com/rdkit/rdkit/issues/1984#issuecomment-409078231
            Chem.GetSymmSSSR(mol)
            
            frag_mol = BRICS.BreakBRICSBonds(mol, sanitize=False)
            frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
            frags = [[atom_idx for atom_idx in frag if atom_idx < mol.GetNumAtoms()] for frag in frags]

            node_feats = {
                "y": data.y,
                "pos": data.pos,
                "z": data.z,
                "charge": data.charge,
            }
            assert mol.GetNumAtoms() == len(node_feats["pos"]), "Fragmentation Error: mol atoms num: {}, dataset atoms num: {}, mol: {}, frag_mol: {}".format(mol.GetNumAtoms(), len(node_feats["pos"]), Chem.MolToSmiles(mol), Chem.MolToSmiles(frag_mol))
            brics_data = build_data(node_feats, frags)

            if self.pre_filter is not None and not self.pre_filter(brics_data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(brics_data)

            data_list.append(brics_data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])
            

             
