__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import itertools
import networkx as nx


def get_production_rule(G, child, itx):

    rhs = nx.Graph()
    for n in G.subgraph(child).nodes():
        rhs.add_node(n)
    for e in G.subgraph(child).edges():
        rhs.add_edge(e[0], e[1])

    # remove links between external nodes (edges in itx)
    for x in itertools.combinations(itx, 2):
        if rhs.has_edge(x[0], x[1]):
            rhs.remove_edge(x[0], x[1])

    return rhs


def add_to_prod_rules(production_rules, lhs, rhs, s):
    prod_rules = production_rules
    letter = 'a'
    d = {}

    for x in lhs:
        d[x] = letter
        letter = chr(ord(letter) + 1)

    lhs_s = set()
    for x in lhs:
        lhs_s.add(d[x])
    if len(lhs_s) == 0:
        lhs_s.add("S")

    i = 0
    rhs_s = nx.Graph()
    for n in rhs.nodes():
        if n in d:
            n = d[n]
        else:
            d[n] = i
            n = i
            i = i + 1
        rhs_s.add_node(n)

    for e in rhs.edges():
        u = d[e[0]]
        v = d[e[1]]
        rhs_s.add_edge(u, v)

    lhs_str = "(" + ",".join(str(x) for x in sorted(lhs_s)) + ")"

    nodes = set()
    rhs_term_dict = []
    for c in sorted(nx.edges(rhs_s)):
        rhs_term_dict.append((",".join(str(x) for x in sorted(list(c))), "T"))
        nodes.add(c[0])
        nodes.add(c[1])

    for c in s:
        rhs_term_dict.append((",".join(str(d[x]) for x in sorted(c)), "N"))
        for x in c:
            nodes.add(d[x])

    for singletons in set(nx.nodes(rhs_s)).difference(nodes):
        rhs_term_dict.append((singletons, "T"))

    rhs_str = ""
    for n in rhs_term_dict:
        rhs_str = rhs_str + "(" + n[0] + ":" + n[1] + ")"
        nodes.add(n[0])
    if rhs_str == "":
        rhs_str = "()"

    if lhs_str not in prod_rules:
        rhs_dict = {}
        rhs_dict[rhs_str] = 1
        prod_rules[lhs_str] = rhs_dict
    else:
        rhs_dict = prod_rules[lhs_str]
        if rhs_str in rhs_dict:
            prod_rules[lhs_str][rhs_str] = rhs_dict[rhs_str] + 1
        else:
            rhs_dict[rhs_str] = 1
            ##sorting above makes rhs match perfectly if a match exists

    print lhs_str, "->", rhs_str


def visit(tu, indent, memo, production_rules, datree, graph):
    G = graph
    T = datree
    prod_rules = production_rules
    if tu in memo:
        return
    memo.add(tu)
    print " " * indent, " ".join(str(x) for x in tu)
    for tv in T[tu]:
        if tv in memo:
            continue
        itx = set(tu).intersection(tv)
        rhs = get_production_rule(G, tv, itx)
        s = list()
        for c in T[tv]:
            if c in memo:  continue
            s.append(list(set(c).intersection(tv)))
        add_to_prod_rules(prod_rules, itx, rhs, s)
        visit(tv, indent + 2, memo, prod_rules, T, G)


def learn_production_rules(G, T):
    prod_rules = {}
    root = list(T)[0]
    rhs = get_production_rule(G, root, set())
    s = list()
    for c in T[root]:
        s.append(list(set(c).intersection(root)))
    add_to_prod_rules(prod_rules, set(), rhs, s)
    visit(root, 0, set(), prod_rules, T, G)
    return prod_rules
