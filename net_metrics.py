__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import collections
from collections import Counter
from random import sample



def draw_ugander_graphlet_plot(orig_g, mG, ergm = [], rmat= []):

    df = pd.DataFrame(mG)
    width=.25
    if len(ergm) > 0:
        dfergm = pd.DataFrame(ergm)
        width=.20

    if len(rmat) > 0:
        rmat = pd.DataFrame(rmat)
        width=.20

    N=11
    dforig = pd.DataFrame(orig_g)
    means = (dforig.mean()['e0'], dforig.mean()['e1'], dforig.mean()['e2'], dforig.mean()['e2c'], dforig.mean()['tri'], dforig.mean()['p3'], dforig.mean()['star'], dforig.mean()['tritail'], dforig.mean()['square'], dforig.mean()['squarediag'], dforig.mean()['k4'])
    sem = (dforig.sem()['e0'], dforig.sem()['e1'], dforig.sem()['e2'], dforig.sem()['e2c'], dforig.sem()['tri'], dforig.sem()['p3'], dforig.sem()['star'], dforig.sem()['tritail'], dforig.sem()['square'], dforig.sem()['squarediag'], dforig.sem()['k4'])
    ind = np.arange(N)
    fig,ax = plt.subplots()
    print means
    rects = ax.bar(ind+.02, means, width-.02, color = 'k', yerr=sem)

    means = (df.mean()['e0'], df.mean()['e1'], df.mean()['e2'], df.mean()['e2c'], df.mean()['tri'], df.mean()['p3'], df.mean()['star'], df.mean()['tritail'], df.mean()['square'], df.mean()['squarediag'], df.mean()['k4'])
    sem = (df.sem()['e0'], df.sem()['e1'], df.sem()['e2'], df.sem()['e2c'], df.sem()['tri'], df.sem()['p3'], df.sem()['star'], df.sem()['tritail'], df.sem()['square'], df.sem()['squarediag'], df.sem()['k4'])
    rects = ax.bar(ind+width+.02, means, width-.02, color = 'b', yerr=sem)
    print means
    ax.set_yscale("log", nonposy='clip')

    if len(ergm) > 0:
        means = (dfergm.mean()['e0'], dfergm.mean()['e1'], dfergm.mean()['e2'], dfergm.mean()['e2c'], dfergm.mean()['tri'], dfergm.mean()['p3'], dfergm.mean()['star'], dfergm.mean()['tritail'], dfergm.mean()['square'], dfergm.mean()['squarediag'], dfergm.mean()['k4'])
        sem = (dfergm.sem()['e0'], dfergm.sem()['e1'], dfergm.sem()['e2'], dfergm.sem()['e2c'], dfergm.sem()['tri'], dfergm.sem()['p3'], dfergm.sem()['star'], dfergm.sem()['tritail'], dfergm.sem()['square'], dfergm.sem()['squarediag'], dfergm.sem()['k4'])
        rects = ax.bar(ind+width+width+width+.02, means, width-.02, color = 'r', yerr=sem)
        print means

    if len(rmat) > 0:
        means = (rmat.mean()['e0'], rmat.mean()['e1'], rmat.mean()['e2'], rmat.mean()['e2c'], rmat.mean()['tri'], rmat.mean()['p3'], rmat.mean()['star'], rmat.mean()['tritail'], rmat.mean()['square'], rmat.mean()['squarediag'], rmat.mean()['k4'])
        print means
        rects = ax.bar(ind+width+width+.02, means, width-.02, color = 'purple')


    plt.ylim(ymin=0)
    #fig = plt.gcf()
    #fig.set_size_inches(5, 3, forward=True)
    plt.show()


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
    width=.25

    N=11
    dforig = pd.DataFrame(orig_g)
    means = (dforig.mean()['e0'], dforig.mean()['e1'], dforig.mean()['e2'], dforig.mean()['e2c'], dforig.mean()['tri'], dforig.mean()['p3'], dforig.mean()['star'], dforig.mean()['tritail'], dforig.mean()['square'], dforig.mean()['squarediag'], dforig.mean()['k4'])
    sem = (dforig.sem()['e0'], dforig.sem()['e1'], dforig.sem()['e2'], dforig.sem()['e2c'], dforig.sem()['tri'], dforig.sem()['p3'], dforig.sem()['star'], dforig.sem()['tritail'], dforig.sem()['square'], dforig.sem()['squarediag'], dforig.sem()['k4'])
    ind = np.arange(N)
    fig,ax = plt.subplots()
    rects = ax.bar(ind+.02, means, width-.02, color = 'k', yerr=sem)

    means = (df.mean()['e0'], df.mean()['e1'], df.mean()['e2'], df.mean()['e2c'], df.mean()['tri'], df.mean()['p3'], df.mean()['star'], df.mean()['tritail'], df.mean()['square'], df.mean()['squarediag'], df.mean()['k4'])
    sem = (df.sem()['e0'], df.sem()['e1'], df.sem()['e2'], df.sem()['e2c'], df.sem()['tri'], df.sem()['p3'], df.sem()['star'], df.sem()['tritail'], df.sem()['square'], df.sem()['squarediag'], df.sem()['k4'])
    rects = ax.bar(ind+width+.02, means, width-.02, color = 'b', yerr=sem)

    plt.ylim(ymin=0)
    #fig = plt.gcf()
    #fig.set_size_inches(5, 3, forward=True)
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



def external_rage(G):
    import subprocess
    import networkx as nx
    from pandas import DataFrame

    with open('tmp.txt', 'w') as tmp:
        for e in G.edges():
            tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')

    args = ("./RAGE.exe", "tmp.txt")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print output

    #results are hardcoded in the exe
    df = DataFrame.from_csv("./Results/UNDIR_RESULTS_tmp.csv", header=0, sep=',', index_col=0)
    df = df.drop('ASType', 1)
    return df


def tijana_eval_rgfd(G_df, H_df):
    T_G = 0.0
    T_H = 0.0
    RGFD = 0.0
    for column in G_df:
        for row in range(0,len(G_df[column])):
            T_G += G_df[column][row]

    for column in H_df:
        for row in range(0,len(H_df[column])):
            T_H += H_df[column][row]

    for column in G_df:
        N_G_i = 0.0
        N_H_i = 0.0
        for row in range(0,len(G_df[column])):
            N_G_i += G_df[column][row]
        for row in range(0,len(H_df[column])):
            N_H_i += H_df[column][row]
        if N_G_i == 0 or N_H_i == 0:
            print 0;
        RGFD += np.log10(N_G_i/T_G) - np.log10(N_H_i/T_H)

    return RGFD

def tijana_eval_compute_gcm(G_df):
    import scipy.stats
    l = len(G_df.columns)
    gcm = np.zeros((l,l))
    i = 0
    for column_G in G_df:
        j=0
        for column_H in G_df:
            gcm[i,j] =  scipy.stats.spearmanr(G_df[column_G].tolist(), G_df[column_H].tolist())[0]
            j+=1
        i+=1
    return gcm

def tijana_eval_compute_gcd(gcm_g, gcm_h):
    import math
    if len(gcm_h) != len(gcm_g):
        raise "Graphs must be same size"
    s = 0
    for i in range(0,len(gcm_g)):
        for j in range(i,len(gcm_h)):
            s += math.pow((gcm_g[i,j] - gcm_h[i,j]), 2)

    gcd = math.sqrt(s)
    return gcd
