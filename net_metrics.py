__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import collections
from collections import Counter
from random import sample


def hops(all_succs, start, level=0, debug=False):
    if debug: print("level:", level)

    succs = all_succs[start] if start in all_succs else []
    if debug: print("succs:", succs)

    lensuccs = len(succs)
    if debug: print("lensuccs:", lensuccs)
    if debug: print()
    if not succs:
        yield level, 0
    else:
        yield level, lensuccs

    for succ in succs:
        # print("succ:", succ)
        for h in hops(all_succs, succ, level + 1):
            yield h


def get_graph_hops(graph, num_samples):
    c = Counter()
    for i in range(0, num_samples):
        node = sample(graph.nodes(), 1)[0]
        b = nx.bfs_successors(graph, node)

        for l, h in hops(b, node):
            c[l] += h

    hopper = Counter()
    for l in c:
        hopper[l] = float(c[l]) / float(num_samples)
    return hopper


def bfs_eff_diam(G, NTestNodes, P):
    EffDiam = -1
    FullDiam = -1
    AvgSPL = -1

    DistToCntH = {}

    NodeIdV = nx.nodes(G)
    random.shuffle(NodeIdV)

    for tries in range(0, min(NTestNodes, nx.number_of_nodes(G))):
        NId = NodeIdV[tries]
        b = nx.bfs_successors(G, NId)
        for l, h in hops(b, NId):
            if h is 0: continue
            if not l + 1 in DistToCntH:
                DistToCntH[l + 1] = h
            else:
                DistToCntH[l + 1] += h

    DistNbrsPdfV = {}
    SumPathL = 0.0
    PathCnt = 0.0
    for i in DistToCntH.keys():
        DistNbrsPdfV[i] = DistToCntH[i]
        SumPathL += i * DistToCntH[i]
        PathCnt += DistToCntH[i]

    oDistNbrsPdfV = collections.OrderedDict(sorted(DistNbrsPdfV.items()))

    CdfV = oDistNbrsPdfV
    for i in range(1, len(CdfV)):
        if not i + 1 in CdfV:
            CdfV[i + 1] = 0
        CdfV[i + 1] = CdfV[i] + CdfV[i + 1]

    EffPairs = P * CdfV[next(reversed(CdfV))]

    for ValN in CdfV.keys():
        if CdfV[ValN] > EffPairs: break

    if ValN >= len(CdfV): return next(reversed(CdfV))
    if ValN is 0: return 1
    # interpolate
    DeltaNbrs = CdfV[ValN] - CdfV[ValN - 1];
    if DeltaNbrs is 0: return ValN;
    return ValN - 1 + (EffPairs - CdfV[ValN - 1]) / DeltaNbrs


def draw_diam_plot(orig_g, mG):
    df = pd.DataFrame(mG)
    gD = bfs_eff_diam(orig_g, 20, .9)
    ori_degree_seq = []
    for i in range(0, len(max(mG))):
        ori_degree_seq.append(gD)

    plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")
    h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
    orig, = plt.plot(ori_degree_seq, color='black', linewidth=2, ls='-', label="H")

    plt.title('Diameter Plot')
    plt.ylabel('Diameter')
    plt.xlabel('Growth')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.legend([orig, h], ['$H$', 'HRG $H^*$'], loc=4)
    # fig = plt.gcf()
    # fig.set_size_inches(5, 4, forward=True)
    plt.show()


def draw_graphlet_plot(orig_g, mG):
    df = pd.DataFrame(mG)
    width = .25

    N = 8
    means = (
        orig_g['e2'], orig_g['t2'], orig_g['t3'], orig_g['q3'], orig_g['qrec'], orig_g['q4'], orig_g['q5'],
        orig_g['q6'])
    ind = np.arange(N)
    fig, ax = plt.subplots()
    rects = ax.bar(ind + .02, means, width - .02, color='k')

    means = (df.mean()['e2'], df.mean()['t2'], df.mean()['t3'], df.mean()['q3'], df.mean()['qrec'], df.mean()['q4'],
             df.mean()['q5'], df.mean()['q6'])
    sem = (
        df.sem()['e2'], df.sem()['t2'], df.sem()['t3'], df.sem()['q3'], df.sem()['qrec'], df.sem()['q4'],
        df.sem()['q5'],
        df.sem()['q6'])
    rects = ax.bar(ind + width + .02, means, width - .02, color='b', yerr=sem)

    plt.title('Graphlet Plot')
    plt.ylabel('Occurrences')
    plt.xlabel('Graphlets')

    plt.ylim(ymin=0)
    # fig = plt.gcf()
    # fig.set_size_inches(5, 4, forward=True)
    plt.show()


def draw_degree_rank_plot(orig_g, mG):
    ori_degree_seq = sorted(nx.degree(orig_g).values(), reverse=True)  # degree sequence
    deg_seqs = []
    for newg in mG:
        deg_seqs.append(sorted(nx.degree(newg).values(), reverse=True))  # degree sequence
    df = pd.DataFrame(deg_seqs)

    plt.xscale('log')
    plt.yscale('log')
    plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")
    h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
    orig, = plt.plot(ori_degree_seq, color='black', linewidth=4, ls='-', label="H")

    plt.title('Degree Distribution')
    plt.ylabel('Degree')
    plt.ylabel('Ordered Vertices')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off

    plt.legend([orig, h], ['$H$', 'HRG $H^*$'], loc=3)
    # fig = plt.gcf()
    # fig.set_size_inches(5, 4, forward=True)
    plt.show()


def draw_network_value(orig_g, mG):
    """
    Network values: The distribution of eigenvector components (indicators of "network value")
    associated to the largest eigenvalue of the graph adjacency matrix has also been found to be
    skewed (Chakrabarti et al., 2004).
    """
    eig_cents = [nx.eigenvector_centrality_numpy(g) for g in mG]  # nodes with eigencentrality

    srt_eig_cents = sorted(eig_cents, reverse=True)
    net_vals = []
    for cntr in eig_cents:
        net_vals.append(sorted(cntr.values(), reverse=True))
    df = pd.DataFrame(net_vals)

    plt.xscale('log')
    plt.yscale('log')
    plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")

    h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
    orig, = plt.plot(sorted(nx.eigenvector_centrality(orig_g).values(), reverse=True), color='black', linewidth=4,
                     ls='-', label="H")

    plt.title('Principle Eigenvector Distribution')
    plt.ylabel('Principle Eigenvector')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off

    plt.legend([orig, h], ['$H$', 'HRG $H^*$'], loc=3)
    # fig = plt.gcf()
    # fig.set_size_inches(5, 4, forward=True)
    plt.show()


def draw_hop_plot(orig_g, mG):
    m_hops_ar = []
    for g in mG:
        c = get_graph_hops(g, 20)
        d = dict(c)
        m_hops_ar.append(d.values())
        print "H* hops finished"

    df = pd.DataFrame(m_hops_ar)

    ## original plot
    c = get_graph_hops(orig_g, 20)
    dorig = dict(c)

    plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")
    h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
    orig, = plt.plot(dorig.values(), color='black', linewidth=4, ls='-', label="H")
    plt.title('Hop Plot')
    plt.ylabel('Reachable Pairs')
    plt.xlabel('Number of Hops')
    # plt.ylim(ymax=max(dorig.values()) + max(dorig.values()) * .10)

    plt.legend([orig, h, ], ['$H$', 'HRG $H^*$'], loc=1)
    #fig = plt.gcf()
    #fig.set_size_inches(5, 4, forward=True)
    plt.show()