import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import torch
from models.CBGNN import CBGNN
from sklearn import metrics
import torch.nn.functional as F
import pickle
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils.khop_intersection import whether_khop_intersection
import time

def train():
    global optimizer
    model.train()
    if optimizer != None:
        optimizer.zero_grad()
    Scores = model(train_Xs, train_Edge_indices, train_Edge2cycles, len_edges, Direction, max_len, use_recurrent_RNN, temperature, whether_k = whether_k)
    Pos_scores = Scores[:pos_split[2]]
    Neg_scores = Scores[pos_split[2]: ]

    # cross entropy loss
    train_y = torch.cat(
        (torch.ones(pos_split[0]), torch.zeros(pos_split[0])))
    if use_cuda:
        train_y = train_y.cuda()
    loss = F.binary_cross_entropy(torch.cat((Pos_scores[:pos_split[0]], Neg_scores[:pos_split[0]])), train_y)


    # margin ranking loss
    #criterion = torch.nn.MarginRankingLoss(loss_margin, reduction='mean')
    #if use_cuda:
    #    loss = criterion(Pos_scores[:pos_split[0]], Neg_scores.view(len(Pos_scores), -1).mean(dim=1)[:pos_split[0]],
    #                 torch.Tensor([1]).cuda()).cuda()
    #else:
    #    loss = criterion(Pos_scores[:pos_split[0]], Neg_scores.view(len(Pos_scores), -1).mean(dim=1)[:pos_split[0]],
    #                     torch.Tensor([1]))


    if optimizer == None:
        total_params = [{'params': model.parameters()}]
        for ccnt_model in range(num_model):
            total_params.append({'params': model.permuteCE[ccnt_model]})
        if learn_combine:
            total_params.append({'params': model.CPMLP, 'lr': learning_rate})
        optimizer = torch.optim.Adam(total_params, lr = learning_rate, weight_decay = weight_decay)

    loss.backward()
    optimizer.step()


    return loss

def test( ):
    model.eval()
    Scores = model(train_Xs, train_Edge_indices, train_Edge2cycles, len_edges, Direction, max_len, use_recurrent_RNN, temperature, whether_k = whether_k)
    Pos_scores = Scores[:pos_split[2]]
    Neg_scores = Scores[pos_split[2]:]

    train_label = [1 for _ in range(pos_split[0])] + [0 for _ in range(pos_split[0])]
    train_scores = Pos_scores.detach().cpu().tolist()[:pos_split[0]] + Neg_scores.detach().cpu().tolist()[:pos_split[0]]

    train_auc = metrics.roc_auc_score(train_label, train_scores)
    train_pr = metrics.average_precision_score(train_label, train_scores)

    val_label = [1 for _ in range(pos_split[1] - pos_split[0])] + [0 for _ in range(pos_split[1] - pos_split[0])]
    val_scores = Pos_scores.detach().cpu().tolist()[pos_split[0]:pos_split[1]] + Neg_scores.detach().cpu().tolist()[
                                                                                 pos_split[0]:pos_split[1]]

    val_auc = metrics.roc_auc_score(val_label, val_scores)
    val_pr = metrics.average_precision_score(val_label, val_scores)

    test_label = [1 for _ in range(pos_split[2] - pos_split[1])] + [0 for _ in range(pos_split[2] - pos_split[1])]
    test_scores = Pos_scores.detach().cpu().tolist()[pos_split[1]:pos_split[2]] + Neg_scores.detach().cpu().tolist()[
                                                                                  pos_split[1]:pos_split[2]]

    test_auc = metrics.roc_auc_score(test_label, test_scores)
    test_pr = metrics.average_precision_score(test_label, test_scores)

    return train_auc, train_pr, val_auc, val_pr, test_auc, test_pr











if __name__ == "__main__":
    #torch.backends.cudnn.enabled = False
    #data_name = "ttest"
    #data_name = "WN18RR_v4"
    #data_name = "fb237_v4"
    data_name = "nell_v4"
    save_model = False
    epochs = 100
    loss_margin = 0.8


    start_time = time.time()


    # for cluster root
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

    if data_name in ['nell_v1']:
        dropout = 0
    else:
        dropout = 0.2
    use_save_file = True
    use_permute = False
    if data_name in ["nell_v3", "nell_v4"]:
        max_len = 6
    elif data_name in ["nell_v2",'WN18RR_v2']:
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
    use_gumbel = True
    learning_rate = 0.005
    weight_decay = 5e-5
    hop = 2
    use_whether_k = False
    use_graph_cluster = True
    
    global optimizer

    if use_inductive:
        from utils.data_utils_inductive_posneg import process_files
        save_pkl_name = "/CBGNN_ind/"
    else:
        # the code is wrong, only inductive is working
        #from utils.data_utils_transductive import process_files
        save_pkl_name = "/CBGNN/"
    if not os.path.exists("./data/" + data_name + save_pkl_name):
        os.mkdir("./data/" + data_name + save_pkl_name)
    num_features = 4 * output_dim * rnn_layers
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


    #seed = np.random.choice(10000)
    seed = 1234
    root = None
    test_root = None
    center_list = None

    for i in range(num_model):
        #seed = np.random.choice(10000)
        if not use_save_file:
            #train_params = process_files(data_name, cycle_the = cycle_the, seed = seed, root = None, num_models = num_model, cnt_model = i, use_graph_cluster = use_graph_cluster, root_list = center_list)
            train_params = process_files(data_name, cycle_the = cycle_the, seed = seed, num_models = num_model, cnt_model = i, use_graph_cluster = use_graph_cluster, pos_root_list = center_list)
            if use_graph_cluster:
                center_list = train_params["pos_root_list"]
        else:
            if os.path.exists("./data/" + data_name + save_pkl_name + str(i) + ".pkl"):
                save_file = open("./data/" + data_name + save_pkl_name + str(i) + ".pkl", "rb")
                print("loading the {}-th data".format(i))
                train_params = pickle.load(save_file)
            else:
                train_params = process_files(data_name, cycle_the = cycle_the, seed = seed, num_models = num_model, cnt_model = i, use_graph_cluster = use_graph_cluster, pos_root_list = center_list)
                if use_graph_cluster:
                    center_list = train_params["pos_root_list"]
                save_file = open("./data/" + data_name + save_pkl_name + str(i) + ".pkl", "wb")
                pickle.dump(train_params, save_file)
                save_file.close()

        #root = train_params["root"]
        if i == 0:
            if use_whether_k:
                whether_k = torch.Tensor(whether_khop_intersection(train_params["pos_triplets"], train_params["neg_triplets"], k = hop))
            else:
                whether_k = None
        if i == 0:
            if learn_features:
                model = CBGNN(num_model, num_features, num_features, len(train_params["relation2id"]), output_dim, rnn_layers, learn_features, learn_combine, dropout, use_permute = use_permute, use_biembed = use_biembed, use_cuda = use_cuda, use_gumbel = use_gumbel)
            else:
                model = CBGNN(num_model, len(train_params["relation2id"]), num_features, None, None, None, learn_features, learn_combine, dropout, use_permute = use_permute, use_biembed = use_biembed, use_cuda = use_cuda, use_gumbel = use_gumbel)

            if use_cuda:
                model = model.cuda()

            #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            optimizer = None
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
            #train_Loops.append(train_params["Path_types"][train_loop_index])

        train_Loops_label.append(train_params["Cycle2positive"][train_loop_index])

        #len_loops = len(train_params["Cycle2edge"])
        len_loops = train_params["len_loops"]
        tmp_edge_index = train_params["edge_index"]
        tmp_edge_index, _ = remove_self_loops(tmp_edge_index)
        tmp_edge_index, _ = add_self_loops(tmp_edge_index, num_nodes=len_loops)


        train_Edge_indices.append(tmp_edge_index)
        #train_Edge2cycles.append(torch.nonzero(train_params["Cycle2edge"].T))
        train_Edge2cycles.append(train_params["Cycle2edge"])
        #len_edges = train_params["Cycle2edge"].size()[1]
        len_edges = train_params["len_edges"]

    start_time_1 = time.time()
    print("Generate SPT cycle bases cost time: {}s".format(start_time_1 - start_time))
    max_val_pr = 0; max_test_pr = 0; max_test_auc = 0; max_val_auc = 0
    for epoch in range(1, epochs + 1):
        if epoch > loop_epoch:
            loop_rate = 0

        train_loss = train()
        if epoch == epochs:
            start_inference = time.time()
        train_auc, train_pr, val_auc, val_pr, test_auc, test_pr = test()
        if epoch == epochs:
            print("Inference a epoch cost time: {}s".format(time.time() - start_inference))
        print("Epoch: {}, Train loss: {}, Train Auc-roc: {}, Train Auc-pr: {}, Val Auc-roc: {}, Val Auc-pr: {}, Test Auc-roc: {}, Test Auc-pr: {}".format(epoch, train_loss, train_auc, train_pr, val_auc, val_pr, test_auc, test_pr))
        if max_val_pr < val_pr:
        #if max_val_auc < val_auc:
            max_val_pr = val_pr
            max_val_auc = val_auc
            max_test_pr = test_pr
            max_test_auc = test_auc
            if save_model:
                torch.save(model.state_dict(), "./data/" + data_name + save_pkl_name + "best_model.pkl")
    if learn_combine:
        print(model.CPMLP)
        print(model.softmax(model.CPMLP / temperature))
    print("Max Val Auc-roc: {}, Max Val Auc-pr: {}, Max Test Auc-roc: {}, Max Test Auc-pr: {}".format(max_val_auc, max_val_pr, max_test_auc, max_test_pr))
    end_time = time.time()
    print("Train and Inference cost time: {}s".format(end_time - start_time_1))
    print("Total cost time: {}s".format(end_time - start_time))
