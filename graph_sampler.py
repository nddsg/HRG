__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import networkx as nx
from random import choice
from collections import deque, Counter


def rwr_sample(G, c, n):
    for i in range(0, c):
        S = choice(G.nodes())

        T = nx.DiGraph()
        T.add_node(S)
        T.add_edges_from(bfs_edges(G, S, n))

        Gprime = nx.subgraph(G, T.nodes())

        yield Gprime


def rwr_sample_depth(G, S, n):
    T = nx.Graph()
    T.add_node(S)
    for a, e in nx.dfs_edges(G, S):
        T.add_edge(a, e)
        if T.number_of_nodes() >= n:
            break;

    Gprime = nx.subgraph(G, T.nodes())

    return Gprime


def quad_graphs():
    qrect = nx.Graph()
    qrect.add_node(0)
    qrect.add_node(1)
    qrect.add_node(2)
    qrect.add_node(3)

    qrect.add_edge(0, 1)
    qrect.add_edge(1, 2)
    qrect.add_edge(2, 3)
    qrect.add_edge(3, 0)

    q4 = nx.Graph()
    q4.add_node(0)
    q4.add_node(1)
    q4.add_node(2)
    q4.add_node(3)

    q4.add_edge(0, 1)
    q4.add_edge(1, 2)
    q4.add_edge(2, 0)
    q4.add_edge(2, 3)

    qstar = nx.Graph()
    qstar.add_node(0)
    qstar.add_node(1)
    qstar.add_node(2)
    qstar.add_node(3)

    qstar.add_edge(0, 1)
    qstar.add_edge(0, 2)
    qstar.add_edge(0, 3)


    return qrect, q4, qstar


def subgraphs_cnt(G, num_smpl):
    sub = Counter()
    sub['e2'] = 0
    sub['t2'] = 0
    sub['t3'] = 0
    sub['q3'] = 0
    sub['q4'] = 0
    sub['qrec'] = 0
    sub['qstar'] = 0
    sub['q5'] = 0
    sub['q6'] = 0
    qrec, q4, qstar = quad_graphs()
    for i in range(0, num_smpl):
        S = choice(G.nodes())
        # size 2
        T = rwr_sample_depth(G, S, 2)
        sub['e2'] += 1

        T = rwr_sample_depth(G, S, 3)
        if T.number_of_nodes() != 3:
            continue
        if T.number_of_edges() == 2:
            sub['t2'] += 1
        else:
            sub['t3'] += 1

        T = rwr_sample_depth(G, S, 4)
        if T.number_of_nodes() != 4:
            continue
        if T.number_of_edges() == 3:
            sub['q3'] += 1
        elif T.number_of_edges() == 4 and nx.is_isomorphic(T, qrec):
            sub['qrec'] += 1
        elif T.number_of_edges() == 4 and nx.is_isomorphic(T, q4):
            sub['q4'] += 1
        elif T.number_of_edges() == 3 and nx.is_isomorphic(T, qstar):
            sub['qstar'] += 1
        elif T.number_of_edges() == 5:
            sub['q5'] += 1
        elif T.number_of_edges() == 6:
            sub['q6'] += 1
        else:
            print "error"

    return sub


def dfs_edges(G, source, n):
    nodes = [source]
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(G[start]))]
        i = 0
        while stack and i < n:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    i += 1
                    yield parent, child
                    visited.add(child)
                    stack.append((child, iter(G[child])))
            except StopIteration:
                stack.pop()


def bfs_edges(G, source, n):
    neighbors = G.neighbors_iter
    visited = set([source])
    queue = deque([(source, neighbors(source))])
    i = 0
    while queue and i < n:
        parent, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                i += 1
                yield parent, child
                visited.add(child)
                queue.append((child, neighbors(child)))
        except StopIteration:
            queue.popleft()
