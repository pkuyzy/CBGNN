import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from scipy.sparse import csc_matrix
from sklearn.cluster import SpectralClustering
import random

def get_dictionary(data_name):
    # get dictionary for the data
    relation2id = {}
    entity2id = {}
    id2relation = {}
    id2entity = {}
    id_relation = 0
    id_entity = 0
    len_original = 0
    for split in ["train", "valid", "test"]:
        if split in ["valid"]:
            split = "train"
            len_original = id_entity
            data_name += "_ind"
        with open("./data/" + data_name + "/" + split + ".txt") as f:
            Lines = f.readlines()
            for line in Lines:
                l = line[:-1].split()
                if l[0] not in entity2id:
                    entity2id[l[0]] = id_entity
                    id2entity[id_entity] = l[0]
                    id_entity += 1
                if l[2] not in entity2id:
                    entity2id[l[2]] = id_entity
                    id2entity[id_entity] = l[2]
                    id_entity += 1
                if l[1] not in relation2id and split == "train":
                    relation2id[l[1]] = id_relation
                    id2relation[id_relation] = l[1]
                    id_relation += 1


    return relation2id, entity2id, id2relation, id2entity, len_original

def get_pos_triplets(relation2id, entity2id, data_name):
    #get all the positive triplets
    pos_triplets = []
    pos_split = []

    l_pos = 0
    for split in ["train", "valid", "test"]:
        if split in ["valid"]:
            split = "train"
            data_name += "_ind"
        with open("./data/" + data_name + "/" + split + ".txt") as f:
            Lines = f.readlines()
            for line in Lines:
                l = line[:-1].split()
                if l[1] in relation2id:
                    pos_triplets.append([entity2id[l[0]], entity2id[l[2]], relation2id[l[1]]])
                    l_pos += 1
            pos_split.append(l_pos)

    adj_list = []
    triplets = np.array(pos_triplets)
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets[:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                             (triplets[:, 0][idx].squeeze(1), triplets[:, 1][idx].squeeze(1))),
                             shape=(len(entity2id), len(entity2id))))
    return pos_triplets, adj_list, pos_split


def sample_neg_triplets(entity2id, pos_triplets, adj_list, pos_split, len_original, neg_sample = 1, seed = 1234):
    #sample the needed negative triplets
    neg_triplets = []
    num_neg = len(pos_triplets) * neg_sample
    cnt = 0
    cnt_seed = 0
    n = len(entity2id)
    import time
    tmp_t = 1000 * time.time()
    #np.random.seed(int(tmp_t) % 2**32)
    np.random.seed(seed)
    choice_seeds_original = np.random.choice(len_original, size=10 * num_neg)
    #choice_seeds_original = np.random.choice(n, size=10 * num_neg)
    #np.random.seed(seed)
    uniform_seeds = np.random.uniform(size = 10 * num_neg)
    choice_seeds_ind = np.random.choice(n - len_original, size = 10 * num_neg) + len_original
    #choice_seeds_ind = np.random.choice(n, size=10 * num_neg)
    while len(neg_triplets) < num_neg:
        cnt_mod = cnt % len(pos_triplets)
        neg_head, neg_tail, rel = pos_triplets[cnt_mod][0], pos_triplets[cnt_mod][1], pos_triplets[cnt_mod][2]

        if cnt_mod  < pos_split[0]:
            if uniform_seeds[cnt_seed] < 0.5:
                neg_head = choice_seeds_original[cnt_seed]
            else:
                neg_tail = choice_seeds_original[cnt_seed]
        else:
            if uniform_seeds[cnt_seed] < 0.5:
                neg_head = choice_seeds_ind[cnt_seed]
            else:
                neg_tail = choice_seeds_ind[cnt_seed]
        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_triplets.append([neg_head, neg_tail, rel])
            cnt += 1
        cnt_seed += 1
    return neg_triplets

def generate_Edge2Relation(pos_triplets, neg_triplets, relation2id):
    Edge2Relation = torch.zeros(len(pos_triplets) + len(neg_triplets), len(relation2id))
    cnt = 0
    for triplet in pos_triplets:
        Edge2Relation[cnt][triplet[2]] = 1
        cnt += 1
    for triplet in neg_triplets:
        Edge2Relation[cnt][triplet[2]] = 1
        cnt += 1
    return Edge2Relation

def make_edge2type(pos_triplets, neg_triplets, pos_split):
    pos_dict_edge2type = {}
    neg_dict_edge2type = {}
    cnt_triplet = 0
    for triplet in pos_triplets:
        edge = (triplet[0], triplet[1])
        new_dict = {}
        if edge not in pos_dict_edge2type:
            new_dict["relation"] = [triplet[2]]
            new_dict["is_tree"] = [0]
            new_dict["is_pos"] = [1]
            new_dict["num"] = [cnt_triplet]
            new_dict["is_train"] = [int(cnt_triplet < pos_split[0])]
            new_dict["is_ind"] = [int(cnt_triplet >= pos_split[1])]
            pos_dict_edge2type[edge] = new_dict
        else:
            pos_dict_edge2type[edge]["relation"].append(triplet[2])
            pos_dict_edge2type[edge]["is_tree"].append(0)
            pos_dict_edge2type[edge]["is_pos"].append(1)
            pos_dict_edge2type[edge]["num"].append(cnt_triplet)
            pos_dict_edge2type[edge]["is_train"].append(int(cnt_triplet < pos_split[0]))
            pos_dict_edge2type[edge]["is_ind"].append(int(cnt_triplet >= pos_split[1]))
        cnt_triplet += 1

    for triplet in neg_triplets:
        edge = (triplet[0], triplet[1])
        new_dict = {}
        if edge not in neg_dict_edge2type:
            new_dict["relation"] = [triplet[2]]
            new_dict["is_tree"] = [0]
            new_dict["is_pos"] = [0]
            new_dict["num"] = [cnt_triplet]
            new_dict["is_train"] = [int(cnt_triplet < pos_split[0] + pos_split[2])]
            new_dict["is_ind"] = [int(cnt_triplet >= pos_split[1] + pos_split[2])]
            neg_dict_edge2type[edge] = new_dict
        else:
            neg_dict_edge2type[edge]["relation"].append(triplet[2])
            neg_dict_edge2type[edge]["is_tree"].append(0)
            neg_dict_edge2type[edge]["is_pos"].append(0)
            neg_dict_edge2type[edge]["num"].append(cnt_triplet)
            neg_dict_edge2type[edge]["is_train"].append(int(cnt_triplet < pos_split[0] + pos_split[2]))
            neg_dict_edge2type[edge]["is_ind"].append(int(cnt_triplet >= pos_split[1] + pos_split[2]))
        cnt_triplet += 1

    return pos_dict_edge2type, neg_dict_edge2type


def make_undirected_graphs_from_data(pos_triplets):
    #generate the undirected graph using all the triplets
    pos_g = nx.Graph()
    for i in pos_triplets:
        pos_g.add_edge(i[0], i[1])
    return pos_g


def select_node(sub_g):
    Nodes = [i for i in sub_g.nodes()]
    return random.choice(Nodes)

def find_loop(dict_edge2type, u, v, node2father, node2tree, root_node, max_loop_len = 4):
    path_u = [u]; path_v = [v]; path_direction_v = []; path_direction_u = []; path_type_u = []; path_type_v = []
    path = []; path_type = []; path_direction = []
    path_num = []; path_num_u = []; path_num_v = []
    path_pos = []; path_pos_u = []; path_pos_v = []
    path_train = []; path_train_u = []; path_train_v = []
    root = root_node[node2tree[u]]
    if root != root_node[node2tree[v]]:
        return [], [], [], [], [], []
    node = u
    cnt_path_len = 0
    while node != root:
        if cnt_path_len > max_loop_len:
            return [], [], [], [], [], []
        edge = (node, node2father[node])
        if edge in dict_edge2type:
            if dict_edge2type[edge]["is_tree"][0] == 1:
                path_direction_u.append(1)
            else:
                edge = (node2father[node], node)
                assert dict_edge2type[edge]["is_tree"][0] == 1
                path_direction_u.append(-1)

        else:
            edge = (node2father[node], node)
            assert  dict_edge2type[edge]["is_tree"][0] == 1
            path_direction_u.append(-1)

        path_u.append(node2father[node])
        path_type_u.append(dict_edge2type[edge]["relation"][0])
        path_num_u.append(dict_edge2type[edge]["num"][0])
        path_pos_u.append(dict_edge2type[edge]["is_pos"][0])
        path_train_u.append(dict_edge2type[edge]["is_train"][0])
        node = node2father[node]
        cnt_path_len += 1

    node = v
    cnt_path_len = 0
    while node != root:
        if cnt_path_len > max_loop_len:
            return [], [], [], [], [], []
        edge = (node, node2father[node])
        if edge in dict_edge2type:
            if dict_edge2type[edge]["is_tree"][0] == 1:
                path_direction_v.append(-1)
            else:
                edge = (node2father[node], node)
                assert dict_edge2type[edge]["is_tree"][0] == 1
                path_direction_v.append(1)
        else:
            edge = (node2father[node], node)
            assert dict_edge2type[edge]["is_tree"][0] == 1
            path_direction_v.append(1)
        path_v.append(node2father[node])
        path_type_v.append(dict_edge2type[edge]["relation"][0])
        path_num_v.append(dict_edge2type[edge]["num"][0])
        path_pos_v.append(dict_edge2type[edge]["is_pos"][0])
        path_train_v.append(dict_edge2type[edge]["is_train"][0])
        node = node2father[node]
        cnt_path_len += 1

    len_u = len(path_u); len_v = len(path_v)
    if len_u > len_v:
        for v_i in range(len_v):
            if path_u[v_i + len_u - len_v] == path_v[v_i]:
                break
        u_i = v_i + len_u - len_v
    else:
        for u_i in range(len_u):
            if path_u[u_i] == path_v[u_i + len_v - len_u]:
                break
        v_i = u_i + len_v - len_u

    path.append(u)
    for i in range(1, u_i + 1):
        path.append(path_u[i])
        path_type.append(path_type_u[i - 1])
        path_direction.append(path_direction_u[i - 1])
        path_num.append(path_num_u[i - 1])
        path_pos.append(path_pos_u[i - 1])
        path_train.append(path_train_u[i - 1])
    for i in range(v_i - 1, -1, -1):
        path.append(path_v[i])
        path_type.append(path_type_v[i])
        path_direction.append(path_direction_v[i])
        path_num.append(path_num_v[i])
        path_pos.append(path_pos_v[i])
        path_train.append(path_train_v[i])

    return path, path_type, path_direction, path_num, path_pos, path_train



def make_matrix_tensor(path, path_type, path_direction, path_num, path_pos, path_train, len_triplets, len_relations):
    # get the torch matrix for future training
    cycle2edge = torch.zeros(len_triplets)
    cycle2relation = torch.zeros(len_relations)
    cycle2pos = 0
    cycle2train = 0
    if len(path) != 0:
        for i in path_num:
            cycle2edge[i] = 1
        for i, j in zip(path_type, path_direction):
            cycle2relation[i] += 1 * j
        cycle2pos = min(path_pos)
        cycle2train = min(path_train)
    return cycle2edge.long(), cycle2relation, cycle2pos, cycle2train


def generate_bfs_tree(pos_g, pos_dict_edge2type, neg_dict_edge2type, len_triplets, len_relations, max_loop_len = 4, pos_root = None):
    edge_type = []
    pos_edge_list = []
    neg_edge_list = []
    pos_root_node = []
    pos_node2tree = {}
    pos_node2father = {}

    #generate the bfs tree for the undirected graph
    cnt_tree = 0
    for sub_c in nx.connected_components(pos_g):
        sub_g = pos_g.subgraph(sub_c)
        if pos_root == None:
            node = select_node(sub_g)
        else:
            node = pos_root[cnt_tree]
        bfs_tree = list(nx.bfs_edges(sub_g, node))
        pos_edge_list += bfs_tree

        pos_root_node.append(node)
        for sub_node in sub_g.nodes():
            pos_node2tree[sub_node] = cnt_tree
        cnt_tree += 1

    #find the tree edges and find father for every nodes
    for cnt in range(len(pos_edge_list)):
        u, v = pos_edge_list[cnt][0], pos_edge_list[cnt][1]
        # find father for every nodes
        pos_node2father[v] = u
        # find the tree edges
        edge = (u, v) if (u, v) in pos_dict_edge2type else (v, u)
        pos_dict_edge2type[edge]["is_tree"][0] = 1


    # compute the length of the matrix
    len_matrix = len_triplets - len(pos_edge_list)

    Cycle2edge = torch.zeros(len_matrix, len_triplets).long()
    Cycle2relation = torch.zeros(len_matrix, len_relations)
    Cycle2positive = torch.zeros(len_matrix)
    train_loop_index = []

    #find all the loop generated by non-tree edges
    print("start find all the loop generated by non-tree edges")
    pbar_matrix = tqdm(total = len_matrix)
    cnt_matrix = 0
    Path_types = []
    Path_directions = []
    previous_ind = 0
    Mark_loop_ind = []
    for edge in pos_dict_edge2type.keys():
        u, v = edge[0], edge[1]
        for relation_cnt in range(len(pos_dict_edge2type[edge]["relation"])):
            if pos_dict_edge2type[edge]["is_tree"][relation_cnt] == 0:
                if pos_dict_edge2type[edge]["is_ind"][relation_cnt] != previous_ind:
                    Mark_loop_ind.append(cnt_matrix)
                previous_ind = pos_dict_edge2type[edge]["is_ind"][relation_cnt]
                path, path_type, path_direction, path_num, path_pos, path_train = find_loop(pos_dict_edge2type, u, v, pos_node2father, pos_node2tree, pos_root_node, max_loop_len)
                if len(path) > 0:
                    path_type.append(pos_dict_edge2type[edge]["relation"][relation_cnt])
                    path_direction.append(-1)
                    path_num.append(pos_dict_edge2type[edge]["num"][relation_cnt])
                    path_pos.append(pos_dict_edge2type[edge]["is_pos"][relation_cnt])
                    path_train.append(pos_dict_edge2type[edge]["is_train"][relation_cnt])
                Cycle2edge[cnt_matrix], Cycle2relation[cnt_matrix], Cycle2positive[cnt_matrix], cycle2train = make_matrix_tensor(path, path_type, path_direction, path_num, path_pos, path_train, len_triplets, len_relations)
                if cycle2train > 0:
                    train_loop_index.append(cnt_matrix)
                Path_types.append(path_type)
                #Path_types.append([path_type[ii] * path_direction[ii] for ii in range(len(path_type))])
                Path_directions.append(path_direction)
                pbar_matrix.update(1)
                cnt_matrix += 1
    for edge in neg_dict_edge2type.keys():
        u, v = edge[0], edge[1]
        for relation_cnt in range(len(neg_dict_edge2type[edge]["relation"])):
            if neg_dict_edge2type[edge]["is_tree"][relation_cnt] == 0:
                if neg_dict_edge2type[edge]["is_ind"][relation_cnt] != previous_ind:
                    Mark_loop_ind.append(cnt_matrix)
                previous_ind = neg_dict_edge2type[edge]["is_ind"][relation_cnt]
                path, path_type, path_direction, path_num, path_pos, path_train = find_loop(pos_dict_edge2type, u, v, pos_node2father, pos_node2tree, pos_root_node, max_loop_len)
                if len(path) > 0:
                    path_type.append(neg_dict_edge2type[edge]["relation"][relation_cnt])
                    path_direction.append(-1)
                    path_num.append(neg_dict_edge2type[edge]["num"][relation_cnt])
                    path_pos.append(neg_dict_edge2type[edge]["is_pos"][relation_cnt])
                    path_train.append(neg_dict_edge2type[edge]["is_train"][relation_cnt])
                Cycle2edge[cnt_matrix], Cycle2relation[cnt_matrix], Cycle2positive[cnt_matrix], cycle2train = make_matrix_tensor(path, path_type, path_direction, path_num, path_pos, path_train, len_triplets, len_relations)
                if cycle2train > 0:
                    train_loop_index.append(cnt_matrix)
                Path_types.append(path_type)
                #Path_types.append([path_type[ii] * path_direction[ii] for ii in range(len(path_type))])
                Path_directions.append(path_direction)
                pbar_matrix.update(1)
                cnt_matrix += 1
    pbar_matrix.close()

    return Cycle2edge, Cycle2relation, Cycle2positive, Path_types, Path_directions, train_loop_index, Mark_loop_ind

def generate_edge_index(Cycle2edge, cycle_the = 3):
    # generate the edge index for CBGNN

    print("start to generate edge index for CBGNN")
    Cycle2edge_T = Cycle2edge.T.float()
    '''
    Cycle_mul = torch.matmul(Cycle2edge.float().cuda(), Cycle2edge_T.cuda()).cpu()
    #edge_index = torch.nonzero(Cycle_mul)
    mask_matrix = (Cycle_mul > cycle_the).long()
    edge_index = torch.nonzero(mask_matrix)
    edge_index = torch.LongTensor(edge_index).T
    print(edge_index.size())
    '''

    from torch_scatter import scatter_add
    Cycle2edge_index = torch.nonzero(Cycle2edge)
    len_loop = len(Cycle2edge)
    out = scatter_add(Cycle2edge_T[Cycle2edge_index[:, 1]], Cycle2edge_index[:, 0], dim = 0, dim_size = len_loop)

    # original one
    #edge_index = torch.nonzero((out > cycle_the).long())
    #edge_index = torch.LongTensor(edge_index).T

    # new one
    topk = torch.topk(out, cycle_the, dim = 1)[1]
    edge_index = torch.LongTensor(2, cycle_the * len(topk))
    for i in range(cycle_the * len(topk)):
        u = int(i / cycle_the); v = int(i % cycle_the)
        edge_index[0, i] = u; edge_index[1, i] = topk[u, v]

    print(edge_index.size())

    return edge_index

def generate_inference_index(Cycle2edge):
    # generate the inference index for CBGNN
    print("start to generate inference index for CBGNN")
    inference_index = []
    pbar_index = tqdm(total=Cycle2edge.size()[1])
    for i in range(Cycle2edge.size()[1]):
        tmp_index = []
        for j in range(Cycle2edge.size()[0]):
            if Cycle2edge[i, j] > 0:
                tmp_index.append(j)
        inference_index.append(tmp_index)
        pbar_index.update(1)
    pbar_index.close()

    return inference_index


def select_node_graph_cluster(g, num_models = 20):
    # need to modify SpectralClustering
    center = []
    for sub_c in nx.connected_components(g):
        sub_g = g.subgraph(sub_c)
        adj_mat = nx.to_numpy_matrix(sub_g)
        n_cluster = num_models if len(sub_g.nodes()) >= num_models else len(sub_g.nodes())
        if len(sub_g.nodes()) < num_models:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        else:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        tmp_center = np.array(np.array(sub_g.nodes())[sc.centers_].tolist() + [random.choice([__ for __ in sub_g.nodes()]) for _ in
                                                                  range(num_models - n_cluster)])#[11:]
        center.append(tmp_center)
    center = np.array(center)
    return center.T

def process_files(data_name, neg_sample = 1, cycle_the = 3, seed = 1234, max_loop_len = 10, pos_root = None, neg_root = None, num_models = 20, cnt_model = 0, use_graph_cluster = False, pos_root_list = None, skip_inference = False):
    params = {}
    params["relation2id"], params["entity2id"], params["id2relation"], params["id2entity"], params["len_original"] = get_dictionary(data_name)
    params["pos_triplets"], params["adj_list"], params["pos_split"] = get_pos_triplets(params["relation2id"], params["entity2id"], data_name)
    params["neg_triplets"] = sample_neg_triplets(params["entity2id"], params["pos_triplets"], params["adj_list"], params["pos_split"], params["len_original"], neg_sample, seed = seed)

    #params["Edge2Relation"] = generate_Edge2Relation(params["pos_triplets"], params["neg_triplets"], params["relation2id"])
    params["pos_dict_edge2type"], params["neg_dict_edge2type"] = make_edge2type(params["pos_triplets"],
                                              params["neg_triplets"], params["pos_split"])
    pos_g = make_undirected_graphs_from_data(params["pos_triplets"])

    # remain some problemes, need to revise
    if use_graph_cluster:
        if cnt_model == 0:
            params["pos_root_list"] = select_node_graph_cluster(pos_g, num_models)
        else:
            params["pos_root_list"] = pos_root_list
    pos_root = pos_root if not use_graph_cluster else params["pos_root_list"][cnt_model].tolist()
    Cycle2edge, params["Cycle2relation"], params["Cycle2positive"], params["Path_types"], params[
        "Path_directions"], params["train_loop_index"], params["mark_loop_ind"] = generate_bfs_tree(pos_g, params["pos_dict_edge2type"], params["neg_dict_edge2type"], len(
        params["pos_triplets"] + params["neg_triplets"]), len(params["relation2id"].keys()), max_loop_len, pos_root = pos_root)
    if not skip_inference:
        params["edge_index"] = generate_edge_index(Cycle2edge, cycle_the=cycle_the)
    params["Cycle2edge"] = torch.nonzero(Cycle2edge.T)
    params["len_loops"] = len(Cycle2edge)
    params["len_edges"] = Cycle2edge.size()[1]
    return params





