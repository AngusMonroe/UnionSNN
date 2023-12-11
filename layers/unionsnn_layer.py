import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class UnionSNNLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """
    def __init__(self, apply_func, aggr_type, dropout, batch_norm, residual=False, init_eps=0, learn_eps=False,
                 e_feat=False, in_dim=None):
        super().__init__()
        self.apply_func = apply_func
        
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
            
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.e_feat = e_feat

        if not in_dim:
            in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.relu = nn.LeakyReLU(0.2, True)
        widths = [1, out_dim]
        self.w_mlp_out = create_wmlp(widths, out_dim, 1)
        self.softmax = nn.Softmax(dim=0)
        
        if in_dim != out_dim:
            self.residual = False
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        self.bn_node_h = nn.BatchNorm1d(out_dim)
        self.message_func = fn.copy_u('h', 'm') if not e_feat else fn.u_mul_e('h', 'w', 'm')

    def forward(self, g, h):
        h_in = h # for residual connection

        g = g.local_var()

        # h = self.relu(self.lin(h))
        h = self.lin(h)
        out_weight = self.w_mlp_out(g.edata['weight'])
        g.edata['w'] = 1 + self.softmax(out_weight)
        g.ndata['h'] = h
        g.update_all(self.message_func, self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        # h = 2 * self.eps * h + g.ndata['neigh']
        if self.apply_func is not None:
            h = self.apply_func(h)

        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
       
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h
    
    
class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


def create_wmlp(widths, nfeato, lbias):
    mlp_modules = []
    for k in range(len(widths) - 1):
        mlp_modules.append(nn.Linear(widths[k], widths[k + 1], bias=False))
        mlp_modules.append(nn.LeakyReLU(0.2, True))
    mlp_modules.append(nn.Linear(widths[len(widths) - 1], nfeato, bias=lbias))
    return nn.Sequential(*mlp_modules)