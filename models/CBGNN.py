#from models.CBGNN_Layer import CBGNN_my
from models.CBGNN_Layer import CBGNN_my
from models.Path_RNN import Path_RNN
import torch
import numpy as np
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
import torch_geometric

class CBGNN(torch.nn.Module):
    def __init__(self, model_num, GNN_input, GNN_output, RNN_input = None, RNN_output = None, RNN_layer = None, learn_feature = True, learn_combine = True, dropout=0.2, use_permute = False, use_biembed = False, use_cuda = True, use_gumbel = False):
        super(CBGNN, self).__init__()
        self.learn_combine = learn_combine
        self.learn_feature = learn_feature
        self.model_num = model_num
        if learn_feature:
            self.RNN = Path_RNN(RNN_input, 20, RNN_output, rnn_layers = RNN_layer, dropout = dropout, use_biembed = use_biembed, use_cuda = use_cuda)
        self.GNN_Models = CBGNN_my(GNN_input, GNN_output, use_permute = use_permute, dropout = dropout) # for CBGNN_Layer_saved, original one
        #self.GNN_Models = CBGNN_my(GNN_input, GNN_output, use_permute=use_permute, dropout=dropout, use_gumbel = use_gumbel)
        self.permuteCE = [None for _ in range(model_num)]
        self.use_cuda = use_cuda
        if learn_combine:
            if use_cuda:
                self.CPMLP = torch.ones(model_num, 1).cuda().requires_grad_()
            else:
                self.CPMLP = torch.ones(model_num, 1).requires_grad_()
            self.softmax = torch.nn.Softmax(dim = 0)


    def forward(self, Features, Edge_indices, Edge2cycles, len_edges, Direction = None, max_len = 6, use_recurrent_RNN = False, temperature = 1, whether_k = None):

        for model_cnt in range(self.model_num):
            if self.learn_feature:
                output = self.RNN.forward(Features[model_cnt], path_direction = Direction[model_cnt], max_len=max_len, use_recurrent_RNN = use_recurrent_RNN)
            else:
                output = Features[model_cnt]
                if self.use_cuda:
                    output = output.cuda()
            edge_index = Edge_indices[model_cnt]

            # add reverse edge which is not included in the original version
            edge_index = torch_geometric.utils.to_undirected(edge_index)

            if self.use_cuda:
                edge_index = edge_index.cuda()
            if self.permuteCE[model_cnt] == None:
                if self.use_cuda:
                    self.permuteCE[model_cnt] = torch.ones(edge_index.size()[1]).cuda().requires_grad_()
                else:
                    self.permuteCE[model_cnt] = torch.ones(edge_index.size()[1]).requires_grad_()
            Edge2cycle = Edge2cycles[model_cnt].long()
            if self.use_cuda:
                Edge2cycle = Edge2cycle.cuda()

            output = self.GNN_Models.encode(output, edge_index)
            #output = output.cpu()

            scores = self.GNN_Models.decode(output, Edge2cycle, edge_index, self.permuteCE[model_cnt], len_edges, whether_k = whether_k)
            if self.use_cuda:
                scores = scores.cuda()

            if not self.learn_combine:
                if model_cnt == 0:
                    Scores = scores
                else:
                    Scores += scores

            else:
                if model_cnt == 0:
                    Scores = scores.reshape(-1, 1)
                else:
                    Scores = torch.cat((Scores, scores.reshape(-1, 1)), dim=1)



        if not self.learn_combine:
            Scores /= self.model_num
            #Scores = torch.clamp(Scores.reshape(-1), 1e-7, 1 - 1e-7)
        else:
            Scores = Scores.mm(self.softmax(self.CPMLP / temperature)).reshape(-1)

        return Scores

    def forward_loop(self, train_Loops, max_len = 6):
        for model_cnt in range(self.model_num):
            if self.learn_feature:
                output = self.RNN.forward(train_Loops[model_cnt], max_len=max_len)
            else:
                output = train_Loops[model_cnt]
                if self.use_cuda:
                    output = output.cuda()
            output = self.GNN_Models.encode_loop(output)
            scores = self.GNN_Models.decode_loop(output)
            if self.use_cuda:
                scores = scores.cuda()

            if model_cnt == 0:
                Scores = scores.reshape(-1)
            else:
                Scores = torch.cat((Scores, scores.reshape(-1)))


        return Scores


def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
        mlp_modules.append(torch.nn.LayerNorm(widths[k + 1]))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)




