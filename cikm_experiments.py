__author__ = 'tweninge'

import networkx as nx
import net_metrics as metrics
import HRG
import PHRG

#import snap

#Rnd = snap.TRnd()
#Graph = snap.GenRMat(1000, 2000, .6, .1, .15, Rnd)
#for EI in Graph.Edges():
#    print "edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())

ba_G = nx.barabasi_albert_graph(500, 3)
er_G = nx.erdos_renyi_graph(500,3000)
sf_G = nx.scale_free_graph(500)
ws_G = nx.watts_strogatz_graph(500,8,.15)
nws_G = nx.newman_watts_strogatz_graph(500,8,.15)

graphs = [ba_G, er_G, sf_G, ws_G, nws_G]


#for G in graphs:
#    chunglu_M = nx.expected_degree_graph(G.degree())
#    HRG_M, degree = HRG.stochastic_hrg(G, 3)
#    pHRG_M = PHRG.probabilistic_hrg(G, 3)



"""
df_h = metrics.external_rage(hstar)
df_g = metrics.external_rage(G)
rgfd = metrics.tijana_eval_rgfd(df_g, df_h)
gcm_g = metrics.tijana_eval_compute_gcm(df_g)
gcm_h = metrics.tijana_eval_compute_gcm(df_h)
gcd = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)


metrics.draw_diam_plot(G, Dstar)
metrics.draw_degree_rank_plot(G, Gstar)
metrics.draw_network_value(G, Gstar)
metrics.draw_hop_plot(G, Gstar)



router_G = nx.read_edgelist("../Phoenix/demo_graphs/as20000102.txt")
enron_G = nx.read_edgelist("../Phoenix/demo_graphs/Email-Enron.txt")
dblp_G = nx.read_edgelist("../Phoenix/demo_graphs/com-dblp.ungraph.txt")
netsci_G = nx.read_edgelist("../Phoenix/demo_graphs/netscience.txt")
power_G = nx.read_edgelist("../Phoenix/demo_graphs/power.txt")
P = [[.9716,.658],[.5684,.1256]] #karate
P = [[.8581,.5116],[.5063,.2071]] #as20000102
#P = [[.7317,.5533],[.5354,.2857]] #dblp
#P = [[.9031,.5793],[.5051,.2136]] #ca-grqc
#P = [[.9124,.5884],[.5029,.2165]] #enron
P = [[.8884,.5908],[.5628,.2736]] #brightkite


kron_M = nx.kronecker_random_graph(k,P,directed=True)

print gcd, rgfd
"""