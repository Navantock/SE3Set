import torch
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.data import Data
from typing import Dict, List, Optional

        
class HyperGraphData(Data):
    """Increase vertex index by data.num_nodes 
    and increase hyperedge index by the number of hyperedges when batch.
    Refer to https://github.com/pyg-team/pytorch_geometric/issues/1195#issuecomment-628020364.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'v2e_index':
            return torch.tensor([[self.num_nodes], [value[1].max().item() + 1]])
        elif key == 'e2v_index':
            return torch.tensor([[value[0].max().item() + 1], [self.num_nodes]])
        else:
            return super(HyperGraphData, self).__inc__(key, value, *args, **kwargs)
        

def build_data(node_feats: Dict[str, Tensor], hyperedges: List[List[int]], he_type: str = "implicit", rc: Optional[float] = None) -> HyperGraphData:
    """Build explicit or implicit hypergraph data with `torch_geometric.data.Data`, whose `edge_index` stores hyperedge as e2v and v2e indices.

    Args:
        node_feats (Dict[str, Tensor]): feature vectors of nodes
        hyperedges (List[List[int]]): hyperedges in the form of a list [[list of nodes in the he]]
        he_type (str, optional): Explicit or implicit hyperedges. Defaults to "implciit".

    Returns:
        HyperGraphData: the hypergraph data for a given hypergraph
    """    
    if he_type == "explicit":
        # E2V indices
        node_list = []
        for he_nodes in hyperedges:
            node_list += he_nodes
        e2v_index = torch.tensor([list(range(len(node_list))), node_list]).long()

        # V2E indices
        v2e_he_indices_repeat, v2e_node_indices = [], []
        for he_nodes in hyperedges:
            cur_size = len(he_nodes)
            v2e_he_indices_repeat += [cur_size] * cur_size
            v2e_node_indices += he_nodes * cur_size
        v2e_he_indices = torch.repeat_interleave(e2v_index[0], torch.tensor(v2e_he_indices_repeat))
        v2e_index = torch.tensor([v2e_node_indices, v2e_he_indices]).long()

        # Edge2Hyperedge Node Position indices
        """ e2he_pos_index = torch.tensor([[src, dst, he] for src, dst, he in zip(e2v_index[1][v2e_he_indices], v2e_node_indices, v2e_he_indices) if src != dst]).long()
        e2he_pos_index.transpose_(0, 1) """

        data = HyperGraphData(v2e_index=v2e_index, e2v_index=e2v_index)

        assert isinstance(node_feats, dict)
        
        for key, value in node_feats.items():
            if isinstance(value, Tensor):
                data[key] = value

        data['num_hyperedges'] = torch.tensor([len(hyperedges)]).long()
        data['num_nw_hyperedges'] = torch.tensor([e2v_index.shape[1]]).long()
        return data
    
    elif he_type == "implicit":
        assert rc is not None, "Radius cutoff for implicit hypergraph is required."
        # origin he indices
        node_list = []
        for he_nodes in hyperedges:
            node_list += he_nodes
        nodewise_he_indices = torch.tensor(list(range(len(node_list)))).long()
        hewise_node_indices = torch.tensor(node_list).long()

        # index each node to its he-level hyperedge
        node_in_e_indices = torch.zeros_like(nodewise_he_indices)
        for i, he_nodes in enumerate(hyperedges):
            node_in_e_indices[he_nodes] = i

        # find neighbors of each node by radius
        assert 'pos' in node_feats.keys(), "Node position is required for E2V indices."
        assert node_feats['pos'].shape[0] == len(node_list), "Node list has different length from node features shape[0], check whether the fragment method generate overlaps between different fragments."
        tmp_index = radius_graph(node_feats["pos"], r=rc, loop=True, max_num_neighbors=1024)
        tmp_index[0] = node_in_e_indices[tmp_index[0]] # node idx to frag-level he idx
        tmp_index = torch.unique(tmp_index, dim=1) # remove duplicate edges
        dst_frag_level_he_index = node_in_e_indices[tmp_index[1]] # node idx to frag-level he idx
        # find the hyperedge which does not contain the node itself
        new_he_index =  torch.where(tmp_index[0] != dst_frag_level_he_index)[0]
        # expand src frag-level he to its nodes
        v2e_src = torch.tensor(sum([hyperedges[i] for i in tmp_index[0]], [])).long()
        # expand dst frag-level he to its repeating the hyperedge index
        node_as_he_indices = torch.zeros_like(hewise_node_indices)
        node_as_he_indices[hewise_node_indices] = nodewise_he_indices
        dst_he_index = node_as_he_indices[tmp_index[1]]
        dst_he_index[new_he_index] = torch.arange(len(new_he_index)).long() + len(nodewise_he_indices)
        v2e_dst = torch.repeat_interleave(dst_he_index, torch.tensor([len(hyperedges[i]) for i in tmp_index[0]])).long()
        

        # v2e index
        v2e_index = torch.stack((v2e_src, v2e_dst), dim=0).long()
        # e2v index
        e2v_index = torch.stack((torch.arange(len(nodewise_he_indices) + len(new_he_index)),
                                torch.cat((hewise_node_indices, tmp_index[1][new_he_index]))), dim=0).long()

        data = HyperGraphData(v2e_index=v2e_index, e2v_index=e2v_index)

        assert isinstance(node_feats, dict)
        
        for key, value in node_feats.items():
            if isinstance(value, Tensor):
                data[key] = value

        data['num_hyperedges'] = torch.tensor([len(hyperedges)]).long()
        data['num_nw_hyperedges'] = torch.tensor([e2v_index.shape[1]]).long()
        return data

    else:
        raise ValueError("Invalid hypergraph type. Choose either 'explicit' or 'implicit'.")
