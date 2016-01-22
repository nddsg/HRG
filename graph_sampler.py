__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import networkx as nx
from random import choice
from collections import deque, Counter



def sample(G):
    S = [0,0,0,0]
    a = choice(G.nodes())
    b = choice(G.nodes())
    c = choice(G.nodes())
    d = choice(G.nodes())

    Gprime = nx.subgraph(G, [a,b,c,d])

    return Gprime


def subgraphs_cnt(G, num_smpl):
    sub = Counter()
    sub['e0'] = 0
    sub['e1'] = 0
    sub['e2'] = 0
    sub['e2c'] = 0
    sub['tri'] = 0
    sub['p3'] = 0
    sub['star'] = 0
    sub['tritail'] = 0
    sub['square'] = 0
    sub['squarediag'] = 0
    sub['k4'] = 0
    for i in range(0,num_smpl):
        #size 2
        T = sample(G)
        #print T.edges()

        if T.number_of_edges() == 0:
            sub['e0'] += 1
        elif T.number_of_edges() == 1:
            sub['e1'] += 1
        elif T.number_of_edges() == 2:
            path = nx.Graph([(0,1), (1,2)])
            if len(max(nx.connected_component_subgraphs(T), key=len)) == 2:
                sub['e2'] += 1
            elif len(max(nx.connected_component_subgraphs(T), key=len)) == 3:
                sub['e2c'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 3:
            #triangle
            triangle = nx.Graph([(0,1), (1,2), (2,0)])
            #path
            path = nx.Graph([(0,1), (1,2), (2,3)])
            #star
            star = nx.Graph([(0,1), (0,2), (0,3)])
            if max(nx.connected_component_subgraphs(T), key=len).number_of_nodes() == 3:
                sub['tri'] += 1
            elif nx.is_isomorphic(T, path):
                sub['p3'] += 1
            elif nx.is_isomorphic(T, star):
                sub['star'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 4:
            square = nx.Graph([(0,1), (1,2), (2,3), (3,0)])
            triangletail = nx.Graph([(0,1), (1,2), (2,0), (2,3)])
            if nx.is_isomorphic(T, square):
                sub['square'] += 1
            elif nx.is_isomorphic(T, triangletail):
                sub['tritail'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 5:
            sub['squarediag'] += 1
        elif T.number_of_edges() == 6:
            sub['k3'] += 1
        else:
            print 'ERROR'

    return sub

