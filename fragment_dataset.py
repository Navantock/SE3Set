import os
import torch
from torch_scatter import scatter
import yaml
import numpy as np
from se3set.data import Hypergraph_QM9, Hypergraph_MD17, Hypergraph_MD22
from tqdm import tqdm

import argparse


Default_Fragmentor_Kwargs = {
    'topoBO_threshold': 1.,
    'min_kernel_atoms_num': 0,
    'min_bo_threshold': 0.1,
    'max_kernel_atoms_num': 8,
    'empr_bo_method': 'SimpleExp',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SE3Set Fragmentation Dataset.')
    parser.add_argument('--train_config', '-c', type=str, help='Train Configuration File')
    args = parser.parse_args()
    
    train_configs = None
    with open(args.train_config, 'r') as f:
        train_configs = yaml.load(f, Loader=yaml.FullLoader)
    assert train_configs is not None, "Invalid Train Configuration File"

    Fragmentor_Kwargs = train_configs['data_params']['fragmentor_kwargs'] if 'fragmentor_kwargs' in train_configs['data_params'] else Default_Fragmentor_Kwargs
    Dataset_Name = train_configs['dataset_name']
    MD_Target_Name = train_configs['data_params']['label']
    Dataset_Path = train_configs['data_params']['root']

    Statistic_Dir = "./train_config/statistics"
    Statistic_Path = os.path.join(Statistic_Dir, Dataset_Name+'.yaml')

    dataset = None
    if "QM9" in Dataset_Name:
        dataset = Hypergraph_QM9(root=Dataset_Path, 
                                 fragmentor_kwargs=Fragmentor_Kwargs)
    elif "MD17" in Dataset_Name:
        dataset = Hypergraph_MD17(root=Dataset_Path,  
                                  fragmentor_kwargs=Fragmentor_Kwargs,
                                  he_type=train_configs['data_params']['he_type'],
                                  implicit_rc=train_configs['data_params']['implicit_rc'],
                                  target=MD_Target_Name)
    elif "MD22" in Dataset_Name:
        dataset = Hypergraph_MD22(root=Dataset_Path, 
                                  fragmentor_kwargs=Fragmentor_Kwargs,
                                  he_type=train_configs['data_params']['he_type'],
                                  implicit_rc=train_configs['data_params']['implicit_rc'],
                                  target=MD_Target_Name)
    # TODO: Add more dataset support
    else:
        raise NotImplementedError('Invalid dataset name: {}'.format(Dataset_Name))

    print("Processing atom types")
    atom_types = torch.cat([torch.unique(data.z) for data in tqdm(dataset)]).unique().tolist()
    print("Processing max z")
    max_z = max([torch.max(data.z).item() for data in tqdm(dataset)])
    print("Processing max distance")
    max_dis_ls = [torch.max(torch.norm(data.pos[data.e2v_index[1][data.v2e_index[1]]] - data.pos[data.v2e_index[0]], dim=-1)) for data in tqdm(dataset)]
    #max_dis_ls = [torch.max(pdist(data.pos)).item() for data in tqdm(dataset)]
    max_dis = float(torch.ceil(max(max_dis_ls)))
    print("Processing hyperedge degree")
    all_he_deg = torch.cat([scatter(torch.ones_like(data.v2e_index[0]), data.v2e_index[1], dim=0) for data in tqdm(dataset)])
    all_avg_he_deg = torch.tensor([torch.mean(scatter(torch.ones_like(data.v2e_index[0]), data.v2e_index[1], dim=0).float()) for data in tqdm(dataset)])
    avg_he_deg = torch.mean(all_he_deg.float()).item()
    max_he_channel = torch.max(all_he_deg).item() - 1
    print("Processing node degree")
    all_node_deg = torch.cat([scatter(torch.ones_like(data.e2v_index[0]), data.e2v_index[1], dim=0) for data in tqdm(dataset)])
    avg_node_deg = torch.mean(all_node_deg.float()).item()
    print("Processing # of atoms")
    n_atoms = [data.z.shape[0] for data in tqdm(dataset)]
    avg_num_nodes = float(np.mean(np.asarray(n_atoms)))
    
    dataset_statistics = {'atom_types': atom_types, 
                          'max_atom_z' : max_z,
                          'avg_num_nodes' : avg_num_nodes,
                          'avg_node_degree' : avg_node_deg,
                          'avg_hyperedge_degree' : avg_he_deg,
                          'hyperedge_channel': max_he_channel,
                          'dataset_name': Dataset_Name, 
                          'max_dis': max_dis,
                          'fragment_kwargs': Fragmentor_Kwargs}
    with open(Statistic_Path, 'w') as f:
        yaml.dump(data=dataset_statistics, stream=f)
    print("Write {} dataset statistics to {}".format(Dataset_Name, Statistic_Path))

    if "MD" in Dataset_Name:
        train_configs["model_params"]["max_radius"] = max_dis
        train_configs["data_params"]["fragmentor_kwargs"] = Fragmentor_Kwargs
        with open(args.train_config, 'w') as f:
            yaml.safe_dump(train_configs, f, indent=2, sort_keys=False)
        print("Update max_radius and fragmentor_kwargs to {} as MD target is used.".format(max_dis))

