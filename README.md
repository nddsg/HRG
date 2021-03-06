# HRG
Hyperedge replacement grammar parser and generator

## Publication

Title: [Growing Graphs from Hyperedge Replacement Graph Grammars](https://arxiv.org/pdf/1608.03192.pdf)
Authors: Salvador Aguiñaga, Rodrigo Palacios†, David Chiang, Tim Weninger University of Notre Dame
†California State University Fresno

### Abstract

Discovering the underlying structures present in large real world graphs is a fundamental scientific problem. In this paper we show that a graph’s clique tree can be used to extract a hyperedge replacement grammar. If we store an ordering from the extraction process, the extracted graph grammar is guaranteed to generate an isomor- phic copy of the original graph. Or, a stochastic application of the graph grammar rules can be used to quickly create random graphs. In experiments on large real world networks, we show that random graphs, generated from extracted graph grammars, exhibit a wide range of properties that are very similar to the original graphs. In addition to graph properties like degree or eigenvector centrality, what a graph “looks like” ultimately depends on small details in lo- cal graph substructures that are difficult to define at a global level. We show that our generative graph model is able to preserve these local substructures when generating new graphs and performs well on new and difficult tests of model robustness.

