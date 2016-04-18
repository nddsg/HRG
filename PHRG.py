__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import os
import re

import networkx as nx

import tree_decomposition as td
import net_metrics as metrics
import graph_sampler as gs
import probabilistic_cfg as pcfg

prod_rules = {}
debug = False


def graph_checks(G):
    ## Target number of nodes
    global num_nodes
    num_nodes = G.number_of_nodes()

    if not nx.is_connected(G):
        print "Graph must be connected";
        os._exit(1)

    if G.number_of_selfloops() > 0:
        print "Graph must be not contain self-loops";
        os._exit(1)


def matcher(lhs, N):
    if lhs == "S":
        return [("S", "S")]
    m = []
    for x in N:
        if len(x) == lhs.count(",") + 1:
            i = 0
            for y in lhs.split(","):
                m.append((x[i], y))
                i += 1
            return m


def binarize(tree):
    (node, children) = tree
    children = [binarize(child) for child in children]
    if len(children) <= 2:
        return (node, children)
    else:
        # Just make copies of node.
        # This is the simplest way to do it, but it might be better to trim unnecessary members from each copy.
        # The order that the children is visited is arbitrary.
        binarized = (node, children[:2])
        for child in children[2:]:
            binarized = (node, [binarized, child])
        return binarized


def grow(rule_list, grammar, diam=0):
    D = list()
    newD = diam
    H = list()
    N = list()
    N.append(["S"])  # starting node
    ttt = 0
    # pick non terminal
    num = 0
    for r in rule_list:
        rule = grammar.by_id[r][0]
        lhs_match = matcher(rule.lhs, N)
        e = []
        match = []
        for tup in lhs_match:
            match.append(tup[0])
            e.append(tup[1])
        lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"

        new_idx = {}
        n_rhs = rule.rhs
        # print lhs_str, "->", n_rhs
        for x in n_rhs:
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
                                    # D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
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
                print n_rhs, new_he
        match = sorted(match)
        N.remove(match)

    newG = nx.Graph()
    for e in H:
        if (len(e) == 1):
            newG.add_node(e[0])
        else:
            newG.add_edge(e[0], e[1])

    return newG, D


def probailistic_hrg(G, num_samples=1):

    graphletG = []

    print G.number_of_nodes()
    print G.number_of_edges()

    G.remove_edges_from(G.selfloop_edges())
    giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.subgraph(G, giant_nodes)

    num_nodes = G.number_of_nodes()

    print G.number_of_nodes()
    print G.number_of_edges()

    graph_checks(G)

    print
    print "--------------------"
    print "-Tree Decomposition-"
    print "--------------------"

    if num_nodes >= 500:
        for Gprime in gs.rwr_sample(G, 2, 100):
            T = td.quickbb(Gprime)
            root = list(T)[0]
            T = td.make_rooted(T, root)
            T = binarize(T)
            root = list(T)[0]
            root, children = T
            td.new_visit(T, G, prod_rules)
    else:
        T = td.quickbb(G)
        root = list(T)[0]
        T = td.make_rooted(T, root)
        T = binarize(T)
        root = list(T)[0]
        root, children = T
        td.new_visit(T, G, prod_rules)

    print
    print "--------------------"
    print "- Production Rules -"
    print "--------------------"

    for k in prod_rules.iterkeys():
        print k
        s = 0
        for d in prod_rules[k]:
            s += prod_rules[k][d]
        for d in prod_rules[k]:
            prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
            print '\t -> ', d, prod_rules[k][d]

    rules = []
    id = 0
    for k, v in prod_rules.iteritems():
        sid = 0
        for x in prod_rules[k]:
            rhs = re.findall("[^()]+", x)
            rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
            print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
            sid += 1
        id += 1

    g = pcfg.Grammar('S')
    for (id, lhs, rhs, prob) in rules:
        g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

    print "Starting max size"

    g.set_max_size(num_nodes)

    print "Done with max size"

    Hstars = []

    for i in range(0, num_samples):
        rule_list = g.sample(num_nodes)
        print rule_list
        hstar = grow(rule_list, g)[0]
        print "H* nodes: " + str(hstar.number_of_nodes())
        print "H* edges: " + str(hstar.number_of_edges())
        Hstars.append(hstar)

    return Hstars



# Example Graphs
# G = nx.star_graph(6)
# G = nx.ladder_graph(10)
# G = nx.karate_club_graph()

G = nx.barabasi_albert_graph(1000,3)
# G = nx.connected_watts_strogatz_graph(200,8,.2)

# G = nx.read_edgelist("../Phoenix/demo_graphs/as20000102.txt")
# G = nx.read_edgelist("../Phoenix/demo_graphs/CA-GrQc.txt")
# G = nx.read_edgelist("../Phoenix/demo_graphs/Email-Enron.txt")
# G = nx.read_edgelist("../Phoenix/demo_graphs/Brightkite_edges.txt")


# Example From KDD Paper
# Graph is undirected
# G = nx.Graph()
# G.add_edge(1, 2)
# G.add_edge(2, 3)
# G.add_edge(2, 4)
# G.add_edge(3, 4)
# G.add_edge(3, 5)
# G.add_edge(4, 6)
# G.add_edge(5, 6)
# G.add_edge(1, 5)

probailistic_hrg(G, 3)