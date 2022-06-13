import numpy as np
import networkx as nx

def whether_khop_intersection(pos_triplets, neg_triplets, k = 2):
    whether_khop = np.zeros(len(pos_triplets) + len(neg_triplets))

    g = nx.Graph()
    for i in pos_triplets:
        g.add_edge(i[0], i[1])
    #for i in neg_triplets:
    #    g.add_edge(i[0], i[1])


    for i in range(len(pos_triplets)):
        u, v, r = pos_triplets[i]
        g.remove_edge(u, v)
        root = u
        nodes_u = [root] + [x for u, x in nx.bfs_edges(g, root, depth_limit=k)]
        root = v
        nodes_v = [root] + [x for v, x in nx.bfs_edges(g, root, depth_limit=k)]
        nodes = list(set(nodes_u) & set(nodes_v))
        if len(nodes) > 0:
            whether_khop[i] = 1
        g.add_edge(u, v)

    for i in range(len(neg_triplets)):
        u, v, r = neg_triplets[i]
        #g.remove_edge(u, v)
        root = u
        nodes_u = [root] + [x for u, x in nx.bfs_edges(g, root, depth_limit=k)]
        root = v
        nodes_v = [root] + [x for v, x in nx.bfs_edges(g, root, depth_limit=k)]
        nodes = list(set(nodes_u) & set(nodes_v))
        if len(nodes) > 0:
            whether_khop[i + len(pos_triplets)] = 1
        #g.add_edge(u, v)


    return whether_khop

