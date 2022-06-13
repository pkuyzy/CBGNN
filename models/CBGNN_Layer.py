import math
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch_geometric
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import SAGEConv, GCNConv, ChebConv

class CBGNN_my(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_permute = False, dropout = 0.2, negative_slope = 0.2):
    #def __init__(self, data, num_features, num_classes, w_mul , dimension=5):
        super(CBGNN_my, self).__init__()
        # setting for CBGNN_layer
        #self.conv1 = CBGNN_layer(out_dim, dropout = dropout, use_gumbel = use_gumbel)
        #self.linear1 = Linear(in_dim, out_dim)
        #self.conv2 = CBGNN_layer(out_dim, dropout = dropout, use_gumbel = use_gumbel)
        #self.linear2 = Linear(out_dim, out_dim)

        # setting for GCNConv
        self.conv1 = GCNConv(in_dim, out_dim, improved = False)
        self.conv2 = GCNConv(out_dim, out_dim, improved = False)

        # setting for SAGEConv
        #self.conv1 = SAGEConv(in_dim, out_dim, normalize = True)
        #self.conv2 = SAGEConv(out_dim, out_dim, normalize = True)

        # setting for ChebNet
        #self.conv1 = ChebConv(in_dim, out_dim, 1)
        #self.conv2 = ChebConv(out_dim, out_dim, 1)

        self.prob_mlp = create_wmlp([out_dim, out_dim], 1, 1)
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.linear_k = create_wmlp([2, out_dim], 1, 1)
        self.use_permute = use_permute

    def encode(self, features, edge_index):
        # to generate node features(default 25 dim)
        x = F.dropout(features, p = self.dropout, training=self.training)

        # setting for CBGNN_layer
        #x = self.linear1(x)

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # setting for CBGNN_layer
        #x = self.linear2(x)

        x = self.conv2(x, edge_index)

        return x

    def encode_loop(self, features):
        x = F.dropout(features, p = self.dropout, training=self.training)

        # setting for CBGNN_layer
        #x = self.linear1(x)

        # setting for GCNConv
        x = torch.matmul(x, self.conv1.weight)
        x += self.conv1.bias

        # setting for SAGEConv
        #x_1 = x
        #x = self.conv1.lin_l(x)
        #x += self.conv1.lin_r(x_1)
        #x =  F.normalize(x, p=2., dim=-1)


        # setting for ChebNet
        #x_1 = x
        #x = torch.matmul(x, self.conv1.weight[0])
        #for k in range(1, self.conv1.weight.size(0)):
        #    x = x + torch.matmul(x_1, self.conv1.weight[k])
        #x += self.conv1.bias

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # setting for CBGNN_layer
        #x = self.linear2(x)

        # setting for GCNConv
        x = torch.matmul(x, self.conv2.weight)
        x += self.conv2.bias

        # setting for SAGEConv
        #x_1 = x
        #x = self.conv2.lin_l(x)
        #x += self.conv2.lin_r(x_1)
        #x = F.normalize(x, p=2., dim=-1)

        # setting for ChebNet
        #x_1 = x
        #x = torch.matmul(x, self.conv2.weight[0])
        #for k in range(1, self.conv2.weight.size(0)):
        #   x = x + torch.matmul(x_1, self.conv2.weight[k])
        #x += self.conv2.bias


        return x

    def decode(self, x, Edge2cycle, edge_index, permuteCE, len_edges, whether_k = None):
        # to induce new relations

        out = self.prob_mlp(x).reshape(-1)

        if self.use_permute:
            #("use_permute")
            permuteCE = F.leaky_relu(permuteCE, self.negative_slope)
            permuteCE = torch_geometric.utils.softmax(permuteCE, edge_index[0])
            permuteCE = F.dropout(permuteCE, p = self.dropout, training = self.training)
            src = out[edge_index[1]] * permuteCE
            out = scatter_add(src, edge_index[0], dim = 0, dim_size = int(edge_index.max()) + 1)

        src = out[Edge2cycle[:, 1]]
        out = scatter_max(src, Edge2cycle[:, 0], dim = 0, dim_size = len_edges)[0]
        #out = scatter_add(src, Edge2cycle[:, 0], dim = 0, dim_size = len_edges)

        if whether_k != None:
            out = self.linear_k(torch.cat((out.reshape(-1, 1), whether_k.reshape(-1, 1).cuda()), dim = 1)).reshape(-1)
        out = F.normalize(out, dim = 0)
        #out = torch.nn.functional.normalize(out)

        #return out
        return 1.0 / (1.0 + torch.exp(-out))

    def decode_loop(self, x):
        out = self.prob_mlp(x).reshape(-1)
        return 1.0 / (1.0 + torch.exp(-out))

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class CBGNN_layer(MessagePassing):
    _alpha: OptTensor

    def __init__(self, out_channels, dropout = 0.0,
                 add_self_loops = True, use_gumbel = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(CBGNN_layer, self).__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.heads = 1
        self.concat = True
        self.negative_slope = 0.2
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.alpha_l = None
        self.use_save_attn = False
        self.use_gumbel = use_gumbel

        self.att_l = Parameter(torch.Tensor(1, 1, out_channels))

        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        zeros(self.bias)


    def forward(self, x, edge_index,
                size = None, return_attention_weights=None, use_save_attn = False):
        H, C = self.heads, x.size()[1]

        self.use_save_attn = use_save_attn

        assert x.dim() == 2
        x_l = x_r = x.view(-1, H, C)
        if not use_save_attn:
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
            self.alpha_l = alpha_l
            self.alpha_r = alpha_r
        else:
            alpha_l = self.alpha_l
            alpha_r = self.alpha_r

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        if self.concat:
            out = out.view(-1, self.heads * C)
        else:
            out = out.mean(dim=1)

        return out


    def message(self, x_j, alpha_j, alpha_i,
                index, ptr,
                size_i):
        if not self.use_save_attn:
            alpha = alpha_j if alpha_i is None else alpha_i + alpha_j
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = torch_geometric.utils.softmax(alpha, index, ptr, size_i)
            self._alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        else:
            alpha = self._alpha
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
        mlp_modules.append(torch.nn.LayerNorm(widths[k+1]))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)