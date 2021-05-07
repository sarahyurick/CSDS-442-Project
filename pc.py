import itertools
from itertools import combinations
from scipy.stats import norm
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder


# see https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf
# for algorithm description


# independence test
# help from https://github.com/Renovamen/pcalg-py
def gauss_ci_test(suffStat, x, y, S):
    C = suffStat["C"]
    n = suffStat["n"]
    cut_at = 0.9999999

    if len(S) == 0:
        r = C[x, y]
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))
    else:
        m = C[np.ix_([x] + [y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1 * cut_at, r))
    # Fisher’s z-transform
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    # Φ^{-1}(1-α/2)
    return 2 * (1 - norm.cdf(abs(res)))


# modeling https://www.rdocumentation.org/packages/pcalg/versions/2.7-1/topics/pc
# Algorithm 1: The PC_pop-algorithm
def pc_pop(suffStat, indepTest, alpha, labels):
    # INPUT: Vertex Set V, Conditional Independence Information
    # OUTPUT: Estimated skeleton C, separation sets S

    # Form the complete undirected graph C on the vertex set V
    sepset = [[[] for _ in range(len(labels))] for _ in range(len(labels))]
    C = [[True for _ in range(len(labels))] for _ in range(len(labels))]
    for i in range(len(labels)):
        C[i][i] = False

    done = False  # done flag

    l: int = 0
    edge_tests = {0: 0}

    # repeat until for each ordered pair of adjacent nodes i,j:
    # |adj(C,i)\{j}| < l
    while not done and any(C) and l <= float("inf"):
        l_new = l + 1
        edge_tests[l_new] = 0
        done = True

        adjacent_pairs = []
        for i in range(len(C)):
            for j in range(len(C[i])):
                if C[i][j]:
                    adjacent_pairs.append((i, j))

        C_copy = C.copy()

        # Select a (new) ordered pair of nodes i,j that are adjacent in C
        # such that |adj(C,i)\{j}| >= l
        # until all ordered pairs of adjacent variables i and j
        # such that |adj(C,i)\{j}| >= l and k contained in adj(C,i)\{j} with |k|=l
        # have been tested for conditional independence
        for i, j in adjacent_pairs:
            if C[i][j]:
                neighborsBool = [row[i] for row in C_copy]
                neighborsBool[j] = False
                # adj(C,i)\{j}
                neighbors = [n for n in range(len(neighborsBool)) if neighborsBool[n]]

                if len(neighbors) >= l:
                    # |adj(C,i)\{j}| > l
                    if len(neighbors) > l:
                        done = False

                    # Choose (new) k contained in adj(C,i)\{j} with |k|=l
                    # until edge i,j is deleted
                    # or all k contained in adj(C,i)\{j} with |k|=l have been chosen

                    # |adj(C,i)\{j}| = l
                    for k in set(itertools.combinations(neighbors, l)):
                        edge_tests[l_new] = edge_tests[l_new] + 1

                        # if i and j are conditionally independent given k
                        p_value = indepTest(suffStat, i, j, list(k))
                        if p_value >= alpha:
                            # Delete edge i,j
                            # Denote this new graph by C
                            C[i][j] = C[j][i] = False
                            # Save k in S(i,j) and S(j,i)
                            sepset[i][j] = list(k)
                            break

        l = l + 1

    return {'skeleton': np.array(C), 'separation set': sepset}


# Algorithm 2: Extending the skeleton to a CPDAG
# help with implementing rules from: https://github.com/Renovamen/pcalg-py/blob/master/pc.py
def extend_cpdag(graph):
    # INPUT: Skeleton G_skel, separation sets S
    G_skel = graph['skeleton']
    S = graph['separation set']

    # OUTPUT: CPDAG G

    # Orient j-k into j->k whenever there is an arrow i->j
    # such that i and k are nonadjacent
    def rule1(pdag):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) \
                        and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)
            if len(isC) > 0:
                for c in isC:
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    # Orient i-j into i->j whenever there is a chain i->k->j
    def rule2(pdag):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) \
                        and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    # Orient i-j into i->j whenever there are two chains i-k->j and i-l->j
    # such that k and l are nonadjacent
    def rule3(pdag):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) \
                        and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    # Rule 4: Orient i-j into i->j whenever there are two chains i-k->l and k->l->j
    # such that k and l are nonadjacent
    pdag = [[0 if not G_skel[i][j] else 1 for i in range(len(G_skel))] for j in range(len(G_skel))]
    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))

    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        allZ = []
        for z in range(len(pdag)):
            if G_skel[y][z] and z != x:
                allZ.append(z)

        for z in allZ:
            if not G_skel[x][z] and S[x][z] is not None and S[z][x] is not None and not (y in S[x][z] or y in S[z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)


# row value -> column value
def get_edges(matrix, names):
    edges = []

    for row in range(len(matrix)):
        for column in range((len(matrix[row]))):
            if matrix[row][column]:
                edges.append((names[row], names[column]))

    return edges


# combines Algorithm 1 and Algorithm 2
def pc(suffStat, alpha, labels, indepTest):
    graph_and_sepset = pc_pop(suffStat, indepTest, alpha, labels)
    cpdag = extend_cpdag(graph_and_sepset)
    return cpdag


if __name__ == '__main__':
    file_path = 'https://raw.githubusercontent.com/sarahyurick/datasets/master/alc_merged.csv'
    image_path = 'pc_DAG.png'

    data = pd.read_csv(file_path)

    drop_col = ['school', 'age', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                'traveltime', 'nursery']
    data = data.drop(columns=drop_col)

    # handle non-numeric features
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        struct_data[col] = le.fit_transform(struct_data[col])

    data = struct_data
    DAG_labels = data.columns

    row_count = sum(1 for row in data)
    p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]},
           alpha=0.05, labels=[str(i) for i in range(row_count)], indepTest=gauss_ci_test)

    # print(p)
    # print(DAG_labels)
    pc_DAG = get_edges(p, DAG_labels)
    print("NO. EDGES:", len(pc_DAG))
    print(pc_DAG)
