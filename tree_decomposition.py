__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

from collections import defaultdict
import itertools
import networkx as nx

def make_clique(graph, nodes):
    for v1 in nodes:
        for v2 in nodes:
            if v1 != v2:
                graph[v1].add(v2)


def is_clique(graph, vs):
    for v1 in vs:
        for v2 in vs:
            if v1 != v2 and v2 not in graph[v1]:
                return False
    return True


def simplicial(graph, v):
    return is_clique(graph, graph[v])


def almost_simplicial(graph, v):
    for u in graph[v]:
        if is_clique(graph, graph[v] - {u}):
            return True
    return False


def eliminate_node(graph, v):
    make_clique(graph, graph[v])
    delete_node(graph, v)


def delete_node(graph, v):
    for u in graph[v]:
        graph[u].remove(v)
    del graph[v]


def copy_graph(graph):
    return {u: set(graph[u]) for u in graph}


def contract_edge(graph, u, v):
    """Contract edge (u,v) by removing u"""
    graph[v] = (graph[v] | graph[u]) - {u, v}
    del graph[u]
    for w in graph:
        if u in graph[w]:
            graph[w] = (graph[w] | {v}) - {u, w}


def upper_bound(graph):
    """Min-fill."""
    graph = copy_graph(graph)
    dmax = 0
    order = []
    while len(graph) > 0:
        # d, u = min((len(graph[u]), u) for u in graph) # min-width
        d, u = min((count_fillin(graph, graph[u]), u) for u in graph)
        dmax = max(dmax, len(graph[u]))
        eliminate_node(graph, u)
        order.append(u)
    return dmax, order


def count_fillin(graph, nodes):
    """How many edges would be needed to make v a clique."""
    count = 0
    for v1 in nodes:
        for v2 in nodes:
            if v1 != v2 and v2 not in graph[v1]:
                count += 1
    return count / 2


def lower_bound(graph):
    """Minor-min-width"""
    graph = copy_graph(graph)
    dmax = 0
    while len(graph) > 0:
        # pick node of minimum degree
        d, u = min((len(graph[u]), u) for u in graph)
        dmax = max(dmax, d)

        # Gogate and Dechter: minor-min-width
        nb = graph[u] - {u}
        if len(nb) > 0:
            _, v = min((len(graph[v] & nb), v) for v in nb)
            contract_edge(graph, u, v)
        else:
            delete_node(graph, u)
    return dmax


class Solution(object):
    pass


def quickbb(graph, fast=True):
    """Gogate and Dechter, A complete anytime algorithm for treewidth. UAI
       2004. http://arxiv.org/pdf/1207.4109.pdf"""

    """Given a permutation of the nodes (called an elimination ordering),
       for each node, remove the node and make its neighbors into a clique.
       The maximum degree of the nodes at the time of their elimination is
       the width of the tree decomposition corresponding to that ordering.
       The treewidth of the graph is the minimum over all possible
       permutations.
       """

    best = Solution()  # this gets around the lack of nonlocal in Python 2
    best.count = 0

    def bb(graph, order, f, g):
        best.count += 1
        if len(graph) < 2:
            if f < best.ub:
                assert f == g
                best.ub = f
                best.order = list(order) + list(graph)

        else:
            vs = []
            for v in graph:
                # very important pruning rule
                if simplicial(graph, v) or almost_simplicial(graph, v) and len(graph[v]) <= lb:
                    vs = [v]
                    break
                else:
                    vs.append(v)

            for v in vs:
                graph1 = copy_graph(graph)
                eliminate_node(graph1, v)
                order1 = order + [v]
                # treewidth for current order so far
                g1 = max(g, len(graph[v]))
                # lower bound given where we are
                f1 = max(g, lower_bound(graph1))
                if f1 < best.ub:
                    bb(graph1, order1, f1, g1)
                    return

    graph = {u: set(graph[u]) for u in graph}

    order = []
    best.ub, best.order = upper_bound(graph)
    lb = lower_bound(graph)

    # This turns on the branch and bound algorithm that
    # gets better treewidth results, but takes a lot
    # longer to process
    if not fast:
        if lb < best.ub:
            bb(graph, order, lb, 0)

    # Build the tree decomposition
    tree = defaultdict(set)

    def build(order):
        if len(order) < 2:
            bag = frozenset(order)
            tree[bag] = set()
            return
        v = order[0]
        clique = graph[v]
        eliminate_node(graph, v)
        build(order[1:])
        for tv in tree:
            if clique.issubset(tv):
                break
        bag = frozenset(clique | {v})
        tree[bag].add(tv)
        tree[tv].add(bag)

    build(best.order)
    return tree


def make_rooted(graph, u, memo=None):
    """Returns: a tree in the format (label, children) where children is a list of trees"""
    if memo is None: memo = set()
    memo.add(u)
    children = [make_rooted(graph, v, memo) for v in graph[u] if v not in memo]
    return (u, children)


def new_visit(datree, graph, prod_rules, indent=0, parent=None):
    G=graph
    node, subtrees = datree

    itx = parent & node if parent else set()
    rhs = get_production_rule(G, node, itx)
    s = [list(node & child) for child, _ in subtrees]
    add_to_prod_rules(prod_rules, itx, rhs, s)

    #print " "*indent, " ".join(str(x) for x in node)
    for subtree in subtrees:
        tv, subsubtrees = subtree
        new_visit(subtree, G, prod_rules, indent=indent+2, parent=node)


def get_production_rule(G, child, itx):

    #lhs = nx.Graph()
    #for n in G.subgraph(itx).nodes():
    #    lhs.add_node(n)
    #for e in G.subgraph(itx).edges():
    #    lhs.add_edge(e[0], e[1])

    rhs = nx.Graph()
    for n in G.subgraph(child).nodes():
        rhs.add_node(n)
    for e in G.subgraph(child).edges():
        rhs.add_edge(e[0], e[1])

    #remove links between external nodes (edges in itx)
    for x in itertools.combinations(itx,2):
        if rhs.has_edge(x[0],x[1]):
            rhs.remove_edge(x[0], x[1])

    #return lhs, rhs
    return rhs


def add_to_prod_rules(production_rules, lhs, rhs, s):
    prod_rules = production_rules
    letter='a'
    d = {}

    for x in lhs:
        d[x]= letter
        letter=chr(ord(letter) + 1)

    lhs_s = set()
    for x in lhs:
        lhs_s.add(d[x])
    if len(lhs_s) == 0:
        lhs_s.add("S")

    i=0
    rhs_s = nx.Graph()
    for n in rhs.nodes():
        if n in d:
            n = d[n]
        else:
            d[n] = i
            n=i
            i=i+1
        rhs_s.add_node(n)

    for e in rhs.edges():
        u = d[e[0]]
        v = d[e[1]]
        rhs_s.add_edge(u,v)


    lhs_str = "(" + ",".join(str(x) for x in sorted(lhs_s)) + ")"

    nodes = set()
    rhs_term_dict = []
    for c in sorted(nx.edges(rhs_s)):
        rhs_term_dict.append( (",".join(str(x) for x in sorted(list(c))), "T") )
        nodes.add(c[0])
        nodes.add(c[1])

    for c in s:
        rhs_term_dict.append( (",".join(str(d[x]) for x in sorted(c)), "N") )
        for x in c:
            nodes.add(d[x])

    for singletons in set(nx.nodes(rhs_s)).difference(nodes):
        rhs_term_dict.append( ( singletons, "T" ) )

    rhs_str=""
    for n in rhs_term_dict:
        rhs_str = rhs_str + "("+n[0]+":"+n[1]+")"
        nodes.add(n[0])
    if rhs_str=="":
        rhs_str = "()"

    if lhs_str not in prod_rules:
        rhs_dict = {}
        rhs_dict[rhs_str] = 1
        prod_rules[lhs_str] = rhs_dict
    else:
        rhs_dict = prod_rules[lhs_str]
        if rhs_str in rhs_dict:
            prod_rules[lhs_str][rhs_str] = rhs_dict[rhs_str]+1
        else:
            rhs_dict[rhs_str] = 1
        ##sorting above makes rhs match perfectly if a match exists

    print lhs_str, "->", rhs_str
