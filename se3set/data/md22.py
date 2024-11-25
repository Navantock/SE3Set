import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import random_split


import numpy as np

from .xyz2mol import xyz2mol
from openff.units import Quantity
from openff.toolkit import Molecule
from .hg_data import build_data
from ..fragment.fragmentor.emprbo_fragmentor import Monomer_EmprBOFragmentor, CappedMonomer_EmprBOFragmentor

from typing import Callable, Optional, Dict, List, Any
from tqdm import tqdm


class LightningMD22Dataset(LightningDataset):
    def __init__(self, 
                 root: str = 'dataset', 
                 label: str = 'aspirin',
                 fragmentor_kwargs: Dict[str, Any]={},
                 he_type: str = 'implicit',
                 implicit_rc: Optional[float] = 5.0,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 frac_list: List[int]=[950, 50], 
                 seed: int=42, 
                 batch_size: int=1, 
                 num_workers: int=0, 
                 **kwargs):
        self.dataset = Hypergraph_MD22(root, fragmentor_kwargs=fragmentor_kwargs, he_type=he_type, implicit_rc=implicit_rc, target=label, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

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

        self.dataset._data.y = self.dataset._data.y.unsqueeze(-1)

        trainset, valset, testset = random_split(self.dataset, frac_list, generator=torch.Generator().manual_seed(seed))

        task_y = self.dataset._data.y[trainset.indices]
        self.task_mean = task_y.mean(dim=0).float()
        self.task_std = task_y.std(dim=0).float()

        super(LightningMD22Dataset, self).__init__(train_dataset=trainset, 
                                                   val_dataset=valset, 
                                                   test_dataset=testset,
                                                   pred_dataset=testset,
                                                   batch_size=batch_size, 
                                                   num_workers=num_workers, 
                                                   **kwargs)
        

class Hypergraph_MD22(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        Ac_Ala3_NHMe='md22_Ac-Ala3-NHMe.npz', 
        Docosahexaenoic_acid='md22_DHA.npz', 
        Stachyose='md22_stachyose.npz', 
        AT_AT='md22_AT-AT.npz', 
        AT_AT_CG_CG='md22_AT-AT-CG-CG.npz', 
        Buckyball_catcher='md22_buckyball-catcher.npz', 
        Double_walled_nanotube='md22_double-walled_nanotube.npz'
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, 
                 root: str, 
                 fragmentor_kwargs: Dict,
                 he_type: str = 'implicit',
                 implicit_rc: Optional[float] = 5.0,
                 target: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.fragmentor_kwargs = fragmentor_kwargs
        self.he_type = he_type
        self.implicit_rc = implicit_rc

        assert target is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(Hypergraph_MD22.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )
        assert target in Hypergraph_MD22.available_molecules + ['all'], "Unknown data argument"

        if target == "all":
            target = ",".join(Hypergraph_MD22.available_molecules)
        self.molecules = target.split(",")

        if len(self.molecules) > 1:
            print(
                "MD22 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(Hypergraph_MD22, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path, weights_only=False)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )
        self.data, self.slices = self.data_all[0], self.slices_all[0]

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(Hypergraph_MD22, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return [Hypergraph_MD22.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md22-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(Hypergraph_MD22.raw_url + file_name, self.raw_dir)

    def process(self):
        fragmentor = Monomer_EmprBOFragmentor(**self.fragmentor_kwargs) if self.he_type == 'implicit' else CappedMonomer_EmprBOFragmentor(**self.fragmentor_kwargs)
        for path in self.raw_paths:
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            rdkit_mol = xyz2mol(z.tolist(), positions[0].tolist(), charge=0)[0]
            for i, (pos, y, dy) in enumerate(tqdm(zip(positions, energies, forces))):
                openff_mol = Molecule.from_rdkit(rdkit_mol)
                openff_mol.conformers[0] = Quantity(pos.numpy(), units='angstrom')
                hyperedges = fragmentor.fragment_index(openff_mol, input_molecule_name="{}".format(i), mode="openff_Molecule", smiles=None, initialize=True, mult=1)

                node_feats = {
                    "y": y,
                    "force": dy,
                    "pos": pos,
                    "z": z
                }
                samples.append(build_data(node_feats=node_feats, hyperedges=hyperedges, he_type=self.he_type, rc=self.implicit_rc))

            if self.pre_filter is not None:
                samples = [data for data in samples if self.pre_filter(data)]

            if self.pre_transform is not None:
                samples = [self.pre_transform(data) for data in samples]

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])