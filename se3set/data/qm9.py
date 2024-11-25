import os, sys
import logging
from typing import Callable, Optional, Dict, List, Any

import torch
from tqdm import tqdm

from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9
from torch_geometric.datasets.qm9 import atomrefs
from torch_geometric.nn.models.schnet import qm9_target_dict

from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import random_split


import numpy as np

from .hg_data import build_data
from .util_units import qm9_label_conversion
from ..fragment.fragmentor.emprbo_fragmentor import Monomer_EmprBOFragmentor, CappedMonomer_EmprBOFragmentor
from ..fragment.moleculeIO.parse import convert_rdkit_mol_to_openff_mol


atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7]  = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8]  = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9]  = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


class LightningQM9Dataset(LightningDataset):
    def __init__(self, 
                 root: str='dataset', 
                 label: str='mu',
                 fragmentor_kwargs: Dict[str, Any]={},
                 he_type: str='explicit',
                 implicit_rc: Optional[float]=None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 frac_list: List[int]=[110000, 10000], 
                 seed: int=42, 
                 batch_size: int=1, 
                 num_workers: int=0, 
                 **kwargs):
        self.dataset = Hypergraph_QM9(root, fragmentor_kwargs=fragmentor_kwargs, he_type=he_type, implicit_rc=implicit_rc, label=label, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        if isinstance(frac_list[0], float):
            if len(frac_list) == 2:
                frac_list += [1. - sum(frac_list)]
            else:
                assert len(frac_list) == 3
        else:
            if len(frac_list) == 2:
                frac_list += [len(self.dataset) - sum(frac_list)]
            else:
                assert len(frac_list) == 3

        trainset, valset, testset = random_split(self.dataset, frac_list, generator=torch.Generator().manual_seed(seed))

        label_idx = self.dataset.label2idx[label]
        task_y = self.dataset._data.y[trainset.indices, label_idx]
        self.task_mean = task_y.mean(dim=0).float()
        self.task_std = task_y.std(dim=0).float()

        super(LightningQM9Dataset, self).__init__(train_dataset=trainset, 
                                                  val_dataset=valset, 
                                                  test_dataset=testset,
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  **kwargs)


class Hypergraph_QM9(QM9):
    def __init__(self, 
                 root: str, 
                 fragmentor_kwargs: Dict,
                 he_type: str = 'explicit',
                 implicit_rc: Optional[float] = None,
                 update_atomrefs: bool = True,
                 label: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        # Attached Fragmentor Arguments
        self.fragmentor_kwargs = fragmentor_kwargs
        self.he_type = he_type
        self.implicit_rc = implicit_rc
        self.processed_info_dir = os.path.join(root, "processed_info")
        self.update_atomrefs = update_atomrefs
        self.label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))

        if label is not None:
            assert label in self.label2idx, (
                "Please pass the desired property to "
                'train on via "label". Available '
                f'properties are {", ".join(self.label2idx)}.'
            )

            self.label = label
            self.label_idx = self.label2idx[self.label]

            if transform is None:
                transform = self._filter_label
            else:
                transform = Compose([transform, self._filter_label])

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            #from rdkit.Chem.rdchem import BondType as BT
            #from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0], weights_only=False)
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * qm9_label_conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        sanitized_suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)

        invalid_idx = []
        fragmentor = Monomer_EmprBOFragmentor(**self.fragmentor_kwargs) if self.he_type == 'implicit' else CappedMonomer_EmprBOFragmentor(**self.fragmentor_kwargs)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            formal_charge = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
                formal_charge.append(atom.GetFormalCharge())

            z = torch.tensor(atomic_number, dtype=torch.long)
            charge = torch.tensor(formal_charge, dtype=torch.long)

            name = mol.GetProp('_Name')

            logging.disable(logging.WARNING)
            openff_mol = convert_rdkit_mol_to_openff_mol(mol, pos)
            try:
                hyperedges = fragmentor.fragment_index(openff_mol, input_molecule_name="{}".format(i), mode="openff_Molecule", smiles=None, initialize=True, mult=1)
            except:
                if sanitized_suppl[i] is None:
                    invalid_idx.append(i)
                    continue
                openff_mol = convert_rdkit_mol_to_openff_mol(sanitized_suppl[i], pos)
                hyperedges = fragmentor.fragment_index(openff_mol, input_molecule_name="{}".format(i), mode="openff_Molecule", smiles=None, initialize=True, mult=1)
            logging.disable(logging.NOTSET)
            
            y = target[i].unsqueeze(0)
            if self.update_atomrefs:
                node_atom = z.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[z]
                atomrefs_value = atomrefs_tensor[node_atom]
                atomrefs_value = torch.sum(atomrefs_value, dim=0, keepdim=True)
                y = y - atomrefs_value  

            node_feats = {
                "y": y,
                "pos": pos,
                "z": z,
                "charge": charge,
                "idx": i,
                "name": name
            }
            data = build_data(node_feats=node_feats, hyperedges=hyperedges, he_type=self.he_type, rc=self.implicit_rc)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        if not os.path.exists(self.processed_info_dir):
            os.makedirs(self.processed_info_dir)
        np.savez(os.path.join(self.processed_info_dir, "invalid.npz"), invalid_idx=np.array(invalid_idx))

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
