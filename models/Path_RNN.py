import numpy as np
import torch


class Path_RNN(torch.nn.Module):
    def __init__(self, relation_num, embedding_dim, output_size, rnn_layers = 2, dropout = 0.5, use_biembed = False, use_cuda = True):
        super(Path_RNN, self).__init__()
        if use_biembed:
            self.embed = torch.nn.Embedding(2 * relation_num + 1, embedding_dim, padding_idx = relation_num)
        else:
            self.embed = torch.nn.Embedding(relation_num + 1, embedding_dim, padding_idx=relation_num)
        self.use_biembed = use_biembed
        self.rnn = torch.nn.LSTM(input_size = embedding_dim, hidden_size = output_size, num_layers = rnn_layers, dropout = dropout, bidirectional = True)
        self.relation_num = relation_num
        self.use_cuda = use_cuda

    def forward(self, path_type, path_direction = None, max_len = 6, use_recurrent_RNN = False):
        large_len = max([len(i) for i in path_type])
        large_len = large_len if large_len <= max_len else max_len
        if self.use_biembed:
            path_type_padding_0, path_type_padding_1 = self.path_type_padding(path_type, path_direction = path_direction, max_len = large_len)
            if self.use_cuda:
                path_type_padding_0, path_type_padding_1 = path_type_padding_0.cuda(), path_type_padding_1.cuda()
        else:
            path_type_padding = self.path_type_padding_original(path_type, max_len = large_len)
            if self.use_cuda:
                path_type_padding = path_type_padding.cuda()

        if self.use_biembed:
            path_embed_0 = self.embed(path_type_padding_0)
            path_embed_1 = self.embed(path_type_padding_1)
            _, (h0, c0) = self.rnn(path_embed_0)
            _, (h1, c1) = self.rnn(path_embed_1)
            h = h0 + h1
            c = c0 + c1
            h = h.permute(1, 0, 2)
            h = h.reshape(len(h), -1)
            c = c.permute(1, 0, 2)
            c = c.reshape(len(h), -1)
            h_c = torch.cat((h, c), dim=1)
        elif not use_recurrent_RNN:
            path_embed = self.embed(path_type_padding)
            output, (h, c) = self.rnn(path_embed)
            h = h.permute(1, 0, 2)
            h = h.reshape(len(h), -1)
            c = c.permute(1, 0, 2)
            c = c.reshape(len(h), -1)
            h_c = torch.cat((h, c), dim=1)
        else:
            for ll in range(large_len):
                index_r = [i for i in range(ll, large_len)] + [i for i in range(ll)]
                path_embed = self.embed(path_type_padding[index_r])
                output, (h, c) = self.rnn(path_embed)
                h = h.permute(1, 0, 2)
                h = h.reshape(len(h), -1)
                c = c.permute(1, 0, 2)
                c = c.reshape(len(h), -1)
                hh = h if ll == 0 else hh + h
                cc = c if ll == 0 else cc + c
            h_c = torch.cat((hh / large_len, c / large_len), dim=1)


        return h_c

    def path_type_padding_original(self, path_type, max_len = 6):
        # path_type is a given list, transform the list to a tensor
        large_len = max_len
        for i in range(len(path_type)):
            while len(path_type[i]) < large_len:
                path_type[i].append(self.relation_num)
            path_type[i] = path_type[i][:large_len]
        path_type = np.array(path_type).T.tolist()
        return torch.LongTensor(path_type)


    def path_type_padding(self, path_type, path_direction, max_len = 6):
        # path_type is a given list, transform the list to a tensor, for nell_v3
        large_len = max_len
        path_type_0 = path_type.copy(); path_type_1 = path_type.copy()
        for i in range(len(path_type)):
            l_path = len(path_type[i])
            if l_path >= 2:
                path_type_new = [path_type[i][l_path - j - 2] for j in range(l_path - 1)] + [path_type[i][l_path - 1]]
                path_direction_new = [path_direction[i][l_path - j - 2] for j in range(l_path - 1)] + [path_direction[i][l_path - 1]]
            else:
                path_type_new = path_type[i]
                path_direction_new = path_direction[i]
            while len(path_type[i]) < large_len:
                path_type[i].append(2 * self.relation_num)
                path_type_new.append(2 * self.relation_num)
                path_direction[i].append(0)
                path_direction_new.append(0)
            path_type[i] = path_type[i][:large_len]
            path_type_new = path_type_new[:large_len]
            path_direction[i] = path_direction[i][:large_len]
            path_direction_new = path_direction_new[:large_len]
            path_type_0[i] = [path_type[i][p] if path_direction[i][p] != -1 else path_type[i][p] + self.relation_num for p in range(large_len)]
            path_type_1[i] = [path_type_new[p] if path_direction_new[p] != 1 else path_type_new[p] + self.relation_num for p in range(large_len)]


        path_type_0 = np.array(path_type_0).T.tolist()
        path_type_1 = np.array(path_type_1).T.tolist()
        return torch.LongTensor(path_type_0), torch.LongTensor(path_type_1)

    def path_type_padding_(self, path_type, path_direction, max_len = 6):
        # path_type is a given list, transform the list to a tensor
        large_len = max_len
        path_type_0 = path_type.copy(); path_type_1 = path_type.copy()
        for i in range(len(path_type)):
            while len(path_type[i]) < large_len:
                path_type[i].append(2 * self.relation_num)
                path_direction[i].append(0)
            path_type[i] = path_type[i][:large_len]
            path_direction[i] = path_direction[i][:large_len]
            path_type_0[i] = [path_type[i][p] if path_direction[i][p] != -1 else path_type[i][p] + self.relation_num for p in range(large_len)]
            path_type_1[i] = [path_type[i][p] if path_direction[i][p] != 1 else path_type[i][p] + self.relation_num for
                              p in range(large_len)]


        path_type_0 = np.array(path_type_0).T.tolist()
        path_type_1 = np.array(path_type_1).T.tolist()
        return torch.LongTensor(path_type_0), torch.LongTensor(path_type_1)



