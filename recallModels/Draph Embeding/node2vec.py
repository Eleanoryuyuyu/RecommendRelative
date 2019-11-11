import numpy as np
import networkx as nx
import random


class Node2Vec():
    def __init__(self, G, is_directed, p, q):
        self.G = G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs)

    def preprocess_transition_probs(self):
        alias_nodes = {}
        for node in self.G.node():
            unnormalized_probs = [self.G[node][nbr]["weight"] for nbr in sorted(self.G.neighbors(node))]
            norm_cost = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_cost for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

    def simulate_walk(self, num_walks, walk_length):
        walks = []
        nodes = list(self.G.nodes())

        for walk_iter in num_walks:
            print(str(walk_iter + 1), '/', str(num_walks))

            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length, node))
        return walks

def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = K * prob
        if q[i] < 1.0:
            smaller.append(q[i])
        else:
            larger.append(q[i])
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop(0)
        large = larger.pop(0)

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(small)
    return J, q


def alias_sample(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
