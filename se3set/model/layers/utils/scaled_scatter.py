from torch import nn
from torch_scatter import scatter


class ScaledScatter(nn.Module):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L693-L706
    """
    def __init__(self, aggregate_num):
        super(ScaledScatter, self).__init__()
        self.aggregate_num = aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.aggregate_num ** 0.5)
        return out    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.aggregate_num)
    