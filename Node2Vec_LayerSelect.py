# Sailung Yeung
# <yeungsl@bu.edu>
# reference:
# https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py

import numpy as np
import random


class Graph():
    def __init__(self, nx_graphs, p, q, r):
        self.G = nx_graphs
        self.p = p
        self.q = q
        self.r = r
        self.jump = 1.0 / len(nx_graphs)

    def node2vec_walk(self, walk_length, start_node, G):
        '''
        Simulate a random walk starting from start node.
        '''
        graphs = self.G
        alias_nodes_list = self.alias_nodes_list
        alias_edges_list = self.alias_edges_list
        alias_jump_list = self.alias_jump_list

        walk = [start_node]
        # print("input graph", graphs.index(G))
        while len(walk) < walk_length:
            # print("pve graph", graphs.index(G))
            cur = walk[-1]
            if isinstance(alias_jump_list[G][cur], tuple):
                # print(cur, alias_jump_list[G][cur])
                G_sub = graphs[alias_draw(alias_jump_list[G][cur][0], alias_jump_list[G][cur][1])]
                '''
                if G_sub != G:
                  print("jumpping to another layer")
                '''
                G = G_sub
            '''
            else:
              print("caught the replica", walk)
            '''
            # print(walk, graphs.index(G))
            cur_nbrs = G.neighbors(cur)
            # print(cur_nbrs)
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes_list[G][cur][0], alias_nodes_list[G][cur][1])])
                else:
                    prev = walk[-2]
                    if (prev, cur) not in alias_edges_list[G].keys():
                        prev = cur_nbrs[alias_draw(alias_nodes_list[G][cur][0], alias_nodes_list[G][cur][1])]
                    next = cur_nbrs[
                        alias_draw(alias_edges_list[G][(prev, cur)][0], alias_edges_list[G][(prev, cur)][1])]
                    walk.append(next)
                    # print("ending graph", graphs.index(G))
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        Gs = self.G
        G = Gs[random.randint(0, len(Gs) - 1)]
        walks = []
        nodes = list(G.nodes())

        # print ('Walk iteration:')
        for walk_iter in range(num_walks):
            # print str(walk_iter+1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node, G=G))

        return walks

    def get_alias_edge(self, src, dst, G):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        Gs = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []

        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        Gs = self.G
        jump = self.jump
        alias_nodes_list = {}
        alias_edges_list = {}
        alias_jump_list = {}
        for G in Gs:
            alias_nodes = {}
            alias_jump = {}
            for node in G.nodes():
                unnormalized_probs = [1 for nbr in sorted(G.neighbors(node))]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                alias_nodes[node] = alias_setup(normalized_probs)
                ######pre-computation of jump######
                jump_list = []
                for graph in Gs:
                    if node in graph.nodes():
                        if graph == G:
                            jump_list.append(self.r)
                        else:
                            jump_list.append((1 / (len(Gs) - 1)) * self.r)
                    else:
                        jump_list.append(0)

                n_con = sum(jump_list)
                n_probs = [float(u_prob) / n_con for u_prob in jump_list]
                alias_jump[node] = alias_setup(n_probs)
                ##### proceed other pre-computation####
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], G)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], G)

            alias_nodes_list[G] = alias_nodes
            alias_edges_list[G] = alias_edges
            alias_jump_list[G] = alias_jump

        self.alias_nodes_list = alias_nodes_list
        self.alias_edges_list = alias_edges_list
        self.alias_jump_list = alias_jump_list

        # print(alias_jump_list)
        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
