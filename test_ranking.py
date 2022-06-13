import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import numpy as np
import torch
from models.CBGNN import CBGNN
from sklearn import metrics
import torch.nn.functional as F
import pickle
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils.khop_intersection import whether_khop_intersection
import time


def test( ):
    model.eval()
    Scores = model(train_Xs, train_Edge_indices, train_Edge2cycles, len_edges, Direction, max_len, use_recurrent_RNN, temperature, whether_k = whether_k)
    Pos_scores = Scores[:pos_split[1]]
    Neg_scores = Scores[pos_split[1]:]


    test_label = [1 for _ in range(pos_split[1] - pos_split[0])] + [0 for _ in range(pos_split[1] - pos_split[0])]
    test_scores = Pos_scores.detach().cpu().tolist()[pos_split[0]:pos_split[1]] + Neg_scores.detach().cpu().tolist()[
                                                                                  pos_split[0]:pos_split[1]]

    #print(len(test_scores))
    #print(len(np.unique(test_scores)))
    #print( len(np.unique(test_scores))/ len(test_scores))
    return test_scores











if __name__ == "__main__":
    #torch.backends.cudnn.enabled = False
    #data_name = "WN18RR_v1"
    data_name = "fb237_v1"
    #data_name = "nell_v1"
    epochs = 100
    loss_margin = 0.8
    if data_name in ["nell_v3", "nell_v4", "fb237_v4", "fb237_v2", "fb237_v1", "WN18RR_v3", "WN18RR_v1"]:
        num_model = 20
    elif data_name in ["nell_v2"]:
        num_model = 50
    else:
        num_model = 30
    # cycle_the = 10
    cycle_the = 2
    use_cuda = False

    learn_features = True
    learn_combine = False
    temperature = 1
    rnn_layers = 2
    output_dim = 20
    if data_name in ["nell_v1"]:
        use_biembed = False
        use_recurrent_RNN = True
    else:
        use_biembed = True
        use_recurrent_RNN = False

    dropout = 0.2
    use_save_file = True
    use_gumbel = False
    if data_name in ["nell_v3", "nell_v4"]:
        max_len = 6 
    elif data_name in ["nell_v2", 'WN18RR_v2']:
        max_len = 8
    elif data_name in ["fb237_v4"]:
        max_len = 7
    elif data_name in ['WN18RR_v3']:
        max_len = 10
    elif data_name in ['nell_v1']:
        max_len = 3
    else:
        max_len = 5
    use_inductive = True
    loop_rate = 0.2
    loop_epoch = 0
    learning_rate = 0.005
    weight_decay = 5e-5
    hop = 2
    use_whether_k = False
    use_graph_cluster = True
    global optimizer

    assert(use_inductive == True)
    if use_inductive:
        from utils.data_utils_inductive_hits import process_files
        save_pkl_name = "/CBGNN_ind/"

    if not os.path.exists("./data/" + data_name + save_pkl_name):
        os.mkdir("./data/" + data_name + save_pkl_name)
    num_features = 4 * output_dim * rnn_layers

    for neg_samp in range(50):
        tmp_t = 1000 * time.time()
        seed = int(tmp_t) % 2**32
        root = None
        test_root = None
        center_list = None
        Models = []
        Model_Xs = []
        Optimizers = []
        train_Xs = []
        Direction = []
        train_Edge_indices = []
        train_Edge2cycles = []
        Len_poses = []
        train_Loops = []
        train_Loops_label = []
        Pos_splits = []
        torch.autograd.set_detect_anomaly(True)
        for i in range(num_model):
            train_params = process_files(data_name, cycle_the=cycle_the, seed=seed, num_models=num_model,
                                             cnt_model=i, use_graph_cluster=use_graph_cluster,
                                             pos_root_list=center_list)
            if use_graph_cluster:
                center_list = train_params["pos_root_list"]

            # root = train_params["root"]
            if i == 0:
                if use_whether_k:
                    whether_k = torch.Tensor(
                        whether_khop_intersection(train_params["pos_triplets"], train_params["neg_triplets"], k=hop))
                else:
                    whether_k = None
            if i == 0:
                if learn_features:
                    model = CBGNN(num_model, num_features, num_features, len(train_params["relation2id"]), output_dim,
                                  rnn_layers, learn_features, learn_combine, dropout,
                                  use_biembed=use_biembed, use_cuda=use_cuda)
                else:
                    model = CBGNN(num_model, len(train_params["relation2id"]), num_features, None, None, None,
                                  learn_features, learn_combine, dropout,
                                  use_biembed=use_biembed, use_cuda=use_cuda)
                model.load_state_dict(torch.load("./data/" + data_name + save_pkl_name + "best_model.pkl"))
                if use_cuda:
                    model = model.cuda()

                len_pos = len(train_params["pos_triplets"])
                pos_split = train_params["pos_split"]

            train_loop_index = train_params["train_loop_index"]
            if not learn_features:
                train_Xs.append(train_params["Cycle2relation"])
                train_Loops.append(train_params["Cycle2relation"][train_loop_index])
            else:
                train_Xs.append(train_params["Path_types"])
                tmp_list = [train_params["Path_types"][loop_index] for loop_index in train_loop_index]
                train_Loops.append(tmp_list)
                Direction.append(train_params["Path_directions"])
                # train_Loops.append(train_params["Path_types"][train_loop_index])

            train_Loops_label.append(train_params["Cycle2positive"][train_loop_index])

            # len_loops = len(train_params["Cycle2edge"])
            len_loops = train_params["len_loops"]
            tmp_edge_index = train_params["edge_index"]
            tmp_edge_index, _ = remove_self_loops(tmp_edge_index)
            tmp_edge_index, _ = add_self_loops(tmp_edge_index, num_nodes=len_loops)

            train_Edge_indices.append(tmp_edge_index)
            # train_Edge2cycles.append(torch.nonzero(train_params["Cycle2edge"].T))
            train_Edge2cycles.append(train_params["Cycle2edge"])
            # len_edges = train_params["Cycle2edge"].size()[1]
            len_edges = train_params["len_edges"]

        test_scores = test()
        if neg_samp == 0:
            total_scores = [test_scores[:pos_split[1] - pos_split[0]]]
        total_scores.append(test_scores[pos_split[1] - pos_split[0]: ])

        del model
        del train_params
    ttt = np.array(total_scores).T

    '''
    # only for evaluation with shuffle
    import random
    indices = np.arange(51)
    #random.shuffle(indices)
    #ind0 = np.where(indices == 0)[0][0]
    #print(ind0)
    #ranks = [np.argwhere(np.argsort(i[indices])[::-1] == ind0) + 1 for i in ttt]
    ranks = []
    for i in ttt:
        random.shuffle(indices)
        ind0 = np.where(indices == 0)[0][0]
        ranks.append(np.argwhere(np.argsort(i[indices])[::-1] == ind0) + 1)
    '''
    ranks = [np.argwhere(np.argsort(i)[::-1] == 0) + 1 for i in ttt] # no random
    hit10list = [x for x in ranks if x <= 10]
    hits_10 = len(hit10list) / len(ranks)
    print("Hits@10 : {}".format(hits_10))
    print(np.shape(ttt))
    #print(total_scores)
