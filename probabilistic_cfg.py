import collections

import networkx as nx
import numpy as np


class Rule(object):
    def __init__(self, id, lhs, rhs, prob, translate=True):
        self.id = id
        self.lhs = lhs
        if translate:
            self.rhs = rhs
            self.cfg_rhs = self.hrg_to_cfg(lhs, rhs)
        else:
            self.cfg_rhs = rhs
        self.prob = prob
        if len(self.cfg_rhs) == 0:
            print "T"

    def hrg_to_cfg(self, lhs, rhs):
        t_symb = set()
        n_symb = []
        for r in rhs:
            if r.endswith(":N"):
                size = [chr(ord('a') + x) for x in range(0, r.count(",") + 1)]
                str = ",".join(size)
                n_symb.append(str)
            else:
                for x in r.split(":")[0].split(","):
                    if x.isdigit(): t_symb.add(x)

        symb = list(t_symb) + n_symb
        return symb


class Grammar(object):
    def __init__(self, start):
        self.start = start
        self.nonterminals = {start}
        self.by_lhs = {}
        self.by_id = {}

    def add_rule(self, rule):
        self.nonterminals.add(rule.lhs)
        self.by_lhs.setdefault(rule.lhs, []).append(rule)
        self.by_id.setdefault(rule.id, []).append(rule)

    def set_max_size(g, max_size):
        unary_graph = nx.DiGraph()
        for x in g.nonterminals:
            unary_graph.add_node(x)
        for lhs, rules in g.by_lhs.items():
            for rule in rules:
                if len(rule.cfg_rhs) == 1 and rule.cfg_rhs[0] in g.nonterminals:
                    unary_graph.add_edge(rule.lhs, rule.cfg_rhs[0], weight=rule.prob)

        try:
            topological = nx.topological_sort(nx.DiGraph(unary_graph), reverse=True)
            unary_matrix = None
        except nx.NetworkXUnfeasible:
            topological = list(g.nonterminals)
            # unary_matrix[i][j] == 1 means there is a rule i->j
            unary_matrix = np.array(nx.to_numpy_matrix(unary_graph, topological))
            # find infinite summation over chains of unary rules (at least one long)
            try:
                unary_matrix = np.dot(unary_matrix, np.array(np.linalg.inv(np.eye(len(topological)) - unary_matrix)))
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(e.message + " (cycle of unary rules with weight >= 1)")

        nt_to_index = {x: i for i, x in enumerate(topological)}

        alpha = np.empty((len(topological), max_size + 1))
        alpha.fill(-np.inf)
        for size in range(1, max_size + 1):
            for lhs_i, lhs in enumerate(topological):
                for rule in g.by_lhs[lhs]:
                    if unary_matrix is not None:
                        # we'll do unary rules later
                        if len(rule.cfg_rhs) == 1 and rule.cfg_rhs[0] in g.nonterminals:
                            continue

                    nts = [nt_to_index[x] for x in rule.cfg_rhs if x in g.nonterminals]
                    n = size - (len(rule.cfg_rhs) - len(nts))  # total size available for nonterminals

                    if len(nts) == 0:
                        if n != 0:
                            continue
                        p = 0.

                    elif len(nts) == 1:
                        p = alpha[nts[0], n]

                    elif len(nts) == 2:
                        if n < 2:
                            continue

                        p = np.logaddexp.reduce([alpha[nts[0], k] + alpha[nts[1], n - k] for k in range(1, n)])
                        # magic sampling may happen here
                        # print "alpha", lhs, n, ' '.join(
                        #    map(str, [np.exp(alpha[nts[0], k] + alpha[nts[1], n - k] - p) for k in range(1, n)]))

                    else:
                        raise ValueError("more than two nonterminals in rhs")

                    with np.errstate(invalid='ignore'):
                        alpha[lhs_i, size] = np.logaddexp(alpha[lhs_i, size], np.log(rule.prob) + p)
            # Apply unary rules
            # If we weren't in log-space, this would be:
            # alpha[:,size] = unary_matrix * alpha[:,size]
            if unary_matrix is not None:
                lz = np.max(alpha[:, size])
                # the reason we made unary_matrix be 1+ applications of unary
                # rules is just in case of underflow here
                alpha[:, size] = np.logaddexp(alpha[:, size],
                                              np.log(np.dot(unary_matrix, np.exp(alpha[:, size] - lz))) + lz)
        g.alpha = alpha
        g.topological = topological

    def sample(g, size):
        w = [(g.start, size)]
        i = 0
        rules = []
        nt_to_index = {x: i for i, x in enumerate(g.topological)}
        while i < len(w):
            if w[i][0] in g.nonterminals:
                lhs, lhs_size = w[i]
                z = g.alpha[nt_to_index[lhs], lhs_size]
                r = np.random.uniform(0, 1., ())
                s = 0.
                for rule in g.by_lhs[lhs]:
                    p = np.log(rule.prob)

                    nts = [nt_to_index[x] for x in rule.cfg_rhs if x in g.nonterminals]
                    n = lhs_size - (len(rule.cfg_rhs) - len(nts))  # total size available for nonterminals

                    if len(nts) == 0:
                        if n != 0:
                            continue
                        s += np.exp(p - z)
                        if s > r:
                            sizes = []
                            break

                    elif len(nts) == 1:
                        s += np.exp(p + g.alpha[nts[0], n] - z)
                        if s > r:
                            sizes = [n]
                            break

                    elif len(nts) == 2:
                        if n < 2:
                            continue
                        for k in xrange(1, n):
                            s += np.exp(p + g.alpha[nts[0], k] + g.alpha[nts[1], n - k] - z)
                            if s > r:
                                sizes = [k, n - k]
                                break
                        if s > r:
                            break

                    else:
                        raise ValueError("more than two nonterminals in rhs")
                else:
                    raise RuntimeError("this shouldn't happen (s=%s)" % (s,))

                rules.append(rule.id)
                cfg_rhs = []
                j = 0
                for y in rule.cfg_rhs:
                    if y in g.nonterminals:
                        cfg_rhs.append((y, sizes[j]))
                        j += 1
                    else:
                        cfg_rhs.append((y, 1))
                w[i:i + 1] = cfg_rhs
            else:
                i += 1
        return rules


if __name__ == "__main__":
    rules = [
        ("r1", "S", ["S", "S"], 0.2),
        ("r2", "S", ["a"], 0.1),
        ("r3", "S", ["T"], 0.7),
        ("r4", "T", ["U"], 1.),
        ("r5", "U", ["S"], 0.9),
        ("r6", "U", ["a"], 0.1),
    ]

    g = Grammar('S')
    for (id, lhs, rhs, prob) in rules:
        g.add_rule(Rule(id, lhs, rhs, prob, False))

    g.set_max_size(6)
    for i in range(10):
        print g.sample(6)
