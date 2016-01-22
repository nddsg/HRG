__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import tree_decomposition as td
import create_production_rules as pr
import graph_sampler as gs
import stochastic_growth as sg
import net_metrics as nm

import networkx as nx

# Example Graphs
# G = nx.star_graph(6)
# G = nx.ladder_graph(10)
G = nx.karate_club_graph()

# G = nx.barabasi_albert_graph(1000,3)
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

# Graph much be connected
if not nx.is_connected(G):
    print "Graph must be connected"
    G = list(nx.connected_component_subgraphs(G))[0]

# Graph must be simple
G.remove_edges_from(G.selfloop_edges())
if G.number_of_selfloops() > 0:
    print "Graph must be not contain self-loops";
    exit()

num_nodes = G.number_of_nodes()
print "Number of Nodes:\t" + str(num_nodes)

num_edges = G.number_of_edges()
print "Number of Edges:\t" + str(num_edges)



# To parse a large graph we use 10 samples of size 500 each. It is
# possible to parse the whole graph, but the approximate
# decomposition method we use is still quite slow.
if num_nodes >= 500:
    for Gprime in gs.rwr_sample(G, 10, 500):
        pr.prod_rules = {}
        T = td.quickbb(Gprime)
        prod_rules = pr.learn_production_rules(Gprime, T)
else:
    T = td.quickbb(G)
    prod_rules = pr.learn_production_rules(G, T)

print "Rule Induction Complete"

Gstar = []
Dstar = []
Gstargl = []
Ggl = []
for run in range(0, 20):
    if num_nodes < 100:
        nG, nD = sg.grow(prod_rules, num_nodes, 1)
    else:
        nG, nD = sg.grow(prod_rules, num_nodes, num_nodes / 50)
    Gstar.append(nG)
    Dstar.append(nD)
    Gstargl.append(gs.subgraphs_cnt(nG, 1000))
    Ggl.append( gs.subgraphs_cnt(G, 1000) )
    print "G* iteration " + str(run) + " of 20"

# Draw figures, ERGM and Kronecker graphs are not included here in this code sample.
nm.draw_graphlet_plot(Ggl, Gstargl)
nm.draw_diam_plot(G, Dstar)
nm.draw_degree_rank_plot(G, Gstar)
nm.draw_network_value(G, Gstar)
nm.draw_hop_plot(G, Gstar)
