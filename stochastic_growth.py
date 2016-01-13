__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import random
import math
import networkx as nx
import net_metrics as metrics

def control_rod(choices, H, num_nodes):
    newchoices = []
    p = len(H) / float(num_nodes)
    total = 0

    for i in range(0, len(choices)):
        n = float(choices[i][0].count('N'))
        t = float(choices[i][0].count('T'))

        # 2*(e^-x)-1
        x = p

        form = 2 * math.e ** (-x) - 1
        wn = n * form
        wt = t

        ratio = max(0, wt + wn)

        total += ratio
        newchoices.append((choices[i][0], ratio))

    r = random.uniform(0, total)
    upto = 0
    if total == 0:
        random.shuffle(newchoices)
    for c, w in newchoices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def try_combination(lhs, N):
    lhs
    for i in range(0, len(N)):
        n = N[i]
        if lhs[0] == "S":
            break
        if len(lhs) == len(n):
            t = []
            for i in n:
                t.append(i)
            random.shuffle(t)
            return zip(t, lhs)
    return []


def find_match(N, prod_rules):
    if len(N) == 1 and ['S'] in N: return [('S', 'S')]
    matching = {}
    while True:
        lhs = random.choice(prod_rules.keys()).lstrip("(").rstrip(")")
        lhs = lhs.split(",")
        c = try_combination(lhs, N)
        if c: return c


def grow(prod_rules, n, diam=0):
    D = list()
    newD = diam
    H = list()
    N = list()
    N.append(["S"])  # starting node
    ttt = 0
    # pick non terminal
    num = 0
    while len(N) > 0 and num < n:
        lhs_match = find_match(N, prod_rules)
        e = []
        match = []
        for tup in lhs_match:
            match.append(tup[0])
            e.append(tup[1])
        lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"

        new_idx = {}
        n_rhs = str(control_rod(prod_rules[lhs_str].items(), H, n)).lstrip("(").rstrip(")")
        # print lhs_str, "->", n_rhs
        for x in n_rhs.split(")("):
            new_he = []
            he = x.split(":")[0]
            term_symb = x.split(":")[1]
            for y in he.split(","):
                if y.isdigit():  # y is internal node
                    if y not in new_idx:
                        new_idx[y] = num
                        num += 1
                        if diam > 0 and num >= newD and len(H) > 0:
                            newD = newD + diam
                            newG = nx.Graph()
                            for e in H:
                                if (len(e) == 1):
                                    newG.add_node(e[0])
                                else:
                                    newG.add_edge(e[0], e[1])
                            D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
                    new_he.append(new_idx[y])
                else:  # y is external node
                    for tup in lhs_match:  # which external node?
                        if tup[1] == y:
                            new_he.append(tup[0])
                            break
            # prod = "(" + ",".join(str(x) for x in new_he) + ")"
            if term_symb == "N":
                N.append(sorted(new_he))
            elif term_symb == "T":
                H.append(new_he)
        match = sorted(match)
        N.remove(match)

    newG = nx.Graph()
    for e in H:
        if (len(e) == 1):
            newG.add_node(e[0])
        else:
            newG.add_edge(e[0], e[1])

    return newG, D
