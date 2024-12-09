# methods for constructing matrices for routing and facility location problems
from util import *
from helper import convert_np_arrays_dc_mp, shortest_path_nodes, k_shortest_paths_nodes, convert_np_arrays_time_dc_mp


def adjacency_matrix_c_mp(G: nx.Graph, paths: list, range_km: float, od_flow_perc: float = 1.0):
    # take in G: undirected graph, paths: list of list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 1. calculate path distance matrix for each path in paths on G
    # 2. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 3. for each (i,j) in path calculate:
    #   i)      a_ij (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # TODO: consider streamlining via matrix form

    mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_' + str(od_flow_perc) +
                                '_p2p_adjacency_mat_mp.pkl')
    if os.path.exists(mat_filepath):
        return pkl.load(open(mat_filepath, 'rb'))

    feasible_paths = []
    for k in range(len(paths)):
        p = paths[k]
        if not nx.has_path(G, source=p[0], target=p[-1]):
            paths.pop(k)
            continue
        p_dists = []
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            p_dists.append(G.edges[i, j]['km'])
        infeas_idx = np.where(np.array(p_dists) > range_km)[0] + 1

        # print(any([set(p).issubset(set(fp)) for fp in feasible_paths]))

        if len(infeas_idx) > 0:
            infeas_idx = np.insert(infeas_idx, 0, 0)
            infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
            for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
                sub_p = p[i:j]
                # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
                if len(sub_p) > 1:
                    feasible_paths.append(sub_p)
        # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
        #     feasible_paths.append(p)
        else:
            feasible_paths.append(p)

    cycle_adj_mats = []
    covered_ods = set()
    for k in range(len(feasible_paths)):
        p = feasible_paths[k]
        if (p[0], p[-1]) not in covered_ods:
            # TODO: do we include the reverse of OD?
            # if OD of this path not yet served
            covered_ods.add((p[0], p[-1]))
        else:
            # OD served by this path already served
            continue

        df = pd.DataFrame(data=0, index=p, columns=p)
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            df.loc[i, j] = G.edges[i, j]['km']
            df.loc[j, i] = df.loc[i, j]
        for i_idx in range(len(p)):
            for j_idx in range(i_idx + 2, len(p)):
                df.loc[p[i_idx], p[j_idx]] = sum([df.loc[p[u], p[u + 1]] for u in range(i_idx, j_idx)])
                df.loc[p[j_idx], p[i_idx]] = df.loc[p[i_idx], p[j_idx]]
        df_a = pd.DataFrame(data=0, index=p, columns=p)
        df_ao = pd.DataFrame(data=0, index=p, columns=p)
        df_an = pd.DataFrame(data=0, index=p, columns=p)
        df_ac = pd.DataFrame(data=0, index=p, columns=p)
        o = p[0]
        n = p[-1]
        for i in p:
            for j in p:
                d = df.loc[i, j]  # i to j via shortest (direct) path
                d_o = df.loc[i, o] + df.loc[o, j]  # i to j via o (0-th node on path)
                d_n = df.loc[i, n] + df.loc[n, j]  # i to j via n (n-th node on path)
                d_c = df.loc[i, n] + df.loc[n, o] + df.loc[o, j]  # i to j via n->o (return path)
                df_a.loc[i, j] = int(d <= range_km)
                df_ao.loc[i, j] = int(d_o <= range_km)
                df_an.loc[i, j] = int(d_n <= range_km)
                df_ac.loc[i, j] = int(d_c <= range_km)

        cycle_adj_mats.append((df_a, df_ao, df_an, df_ac))

    with open(mat_filepath, 'wb') as f:
        pkl.dump(cycle_adj_mats, f)
        f.close()

    return cycle_adj_mats


def constraint_matrix_dc_mp(G: nx.Graph, ods: list, node_list: list, range_km: float, od_flow_perc: float = 1):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    # # START CORRIDOR TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_deviation_paths.pkl')
    # # END CORRIDOR TEST - OUTCOMMENT BELOW
    # # START RANDOM TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                             'random_s1000_n500_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                                   'random_s1000_n500_deviation_paths.pkl')
    # # END RANDOM TEST - OUTCOMMENT BELOW
    mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_'
                                + str(od_flow_perc) + '_ifc_constraint_mat_mp.npz')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_'
    #                                   + str(od_flow_perc) + '_ifc_deviation_paths_mp.pkl')
    if os.path.exists(mat_filepath):
        return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True))

    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]

    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # for k in range(len(paths)):
    #     p = paths[k]
    #     if not nx.has_path(G, source=p[0], target=p[-1]):
    #         paths.pop(k)
    #         continue
    #     p_dists = []
    #     for i, j in zip(p[:-1], p[1:]):
    #         # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
    #         p_dists.append(G.edges[i, j]['km'])
    #     infeas_idx = np.where(np.array(p_dists) > D)[0] + 1
    #
    #     if len(infeas_idx) > 0:
    #         infeas_idx = np.insert(infeas_idx, 0, 0)
    #         infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
    #         for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
    #             sub_p = p[i:j]
    #             # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
    #             if len(sub_p) > 1:
    #                 feasible_paths.append(sub_p)
    #     # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
    #     #     feasible_paths.append(p)
    #     else:
    #         feasible_paths.append(p)

    # 3. generate set of paths (incl. super paths) for each O-D pair
    # ods = [(o, d) for o, d in ods if (o, d) in od_flows.keys() and od_flows[o, d] > 0]
    # paths = {(p[0], p[-1]): p for p in paths if (p[0], p[-1]) in ods}
    # ods = [(o, d) for o, d in ods if ((o, d) in od_flows.keys() and od_flows[o, d] > 0) or
    #        ((d, o) in od_flows.keys() and od_flows[d, o] > 0)]
    paths = {(o, d): shortest_path_nodes(G, source=o, target=d, weight='km') for o, d in ods}
    print('ODs %d  Paths %d' % (len(ods), len(paths)))
    deviation_paths = {od: [p] for od, p in paths.items()}
    # deviation_paths_node_lists = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                # get base path for O-D pair (k, l)
                p_kl = paths[k, l]
                # if i and j on base path for O-D pair (k, l)
                if i in p_kl and j in p_kl:
                    i_idx = p_kl.index(i)
                    j_idx = p_kl.index(j)
                    # if j comes after i on this path
                    if i_idx < j_idx:
                        # orient superpath: i->k + k+1->l-1 + l->j
                        # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                        p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                        # add this new path
                        deviation_paths[i, j].append(p_iklj)
                        # deviation_paths_node_lists[i, j].append(p_kl)
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                    # if i comes after j on this path
                    else:
                        # orient superpath the opposite way: i->l + l->k + k->j
                        p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                        # add this new path
                        deviation_paths[i, j].append(p_ilkj)
                        # deviation_paths_node_lists[i, j].append(list(reversed(p_kl)))
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_ilkj)))

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    for i, j in unique_ods:
        deviation_paths[i, j] = [list(q) for q in set(tuple(p) for p in deviation_paths[i, j])]
        deviation_paths[j, i] = [list(reversed(p)) for p in deviation_paths[i, j]]

    print(len(deviation_paths))
    print(sum(len(deviation_paths[od]) for od in deviation_paths.keys()))

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    # O-D direction
    ods_list = list(unique_ods)
    node_path_od_comb = [(i, phi, od) for od in ods_list for phi in range(len(deviation_paths[od]))
                         for i in deviation_paths[od][phi]]
    npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    path_od_comb = [(p, od) for od in ods_list for p in range(len(deviation_paths[od]))]
    pod_comb_idx = {v: i for i, v in enumerate(path_od_comb)}
    # D-O (reverse) direction
    dos_list = [(d, o) for o, d in ods_list]
    node_path_do_comb = [(i, phi, do) for do in dos_list for phi in range(len(deviation_paths[do]))
                         for i in deviation_paths[do][phi]]
    npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    path_do_comb = [(p, do) for do in dos_list for p in range(len(deviation_paths[do]))]
    pdo_comb_idx = {v: i for i, v in enumerate(path_do_comb)}

    n = len(node_list)
    node_idx = {v: i for i, v in enumerate(node_list)}
    # path_od_dists = {(p, od): -1 for od in ods for p in range(len(deviation_paths[od]))}
    path_od_dists = {(p, od): [] for od, p_idxs in deviation_paths.items() for p in range(len(p_idxs))}
    deviation_paths_node_lists = {od: {p: [] for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    for o, d in deviation_paths_node_lists.keys():
        for p in deviation_paths_node_lists[o, d].keys():
            dp = np.array(deviation_paths[o, d][p])
            # can have repeat at start, end, or both
            o_idx = np.where(dp == o)[0]
            d_idx = np.where(dp == d)[0]
            if len(o_idx) == 1 and len(d_idx) == 1:
                deviation_paths_node_lists[o, d][p] = dp.tolist()
            elif len(o_idx) > 1 and len(d_idx) == 1:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[dp_o_idx:].tolist()
            elif len(o_idx) == 1 and len(d_idx) > 1:
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[:-dp_d_idx].tolist()
            else:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[dp_o_idx:-dp_d_idx].tolist()

    # deviation_paths_node_lists = {od: {p: deviation_paths[od][p][:len(set(deviation_paths[od][p]))]
    #                                    for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    # deviation_paths_node_lists = {od: {p: list(set(deviation_paths[od][p]))
    #                                    for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    deviation_paths_node_idxs = {od: {p: {v: i for i, v in enumerate(deviation_paths_node_lists[od][p])}
                                      for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}

    # suppose have: od = (a, b) so that do = (b, a) and have dp[od][1] = [a,1,b,2,3,4,3,2,b] (a->1->b<->2<->3<->4)
    # so that it is cleaned to dpnl[od][1] = [a,1,b,2,3,4] and
    # dp[do][1] = [b,2,3,4,3,2,b,1,a] -> dpnl[do][1] = [4,3,2,b,1,a]
    # now we have: a_idx_phi = 0, 1_idx_phi = 1, b_idx_phi = 2, ...
    # a_idx_rho = 5, 1_idx_phi = 4, b_idx_phi = 3, ...

    # TODO: check calculations
    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            if len(path_od_dists[phi_idx, od]) == 0:
                phi_nodes = deviation_paths_node_lists[od][phi_idx]
                phi_node_idx = deviation_paths_node_idxs[od][phi_idx]
                d_phi = np.zeros((len(phi_nodes), len(phi_nodes)))
                # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
                for i, j in zip(phi[:-1], phi[1:]):
                    i_idx_phi = phi_node_idx[i]
                    j_idx_phi = phi_node_idx[j]
                    if d_phi[i_idx_phi, j_idx_phi] == 0:
                        d_phi[i_idx_phi, j_idx_phi] = G.edges[i, j]['km']
                        d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]  # reverse
                for i_idx in range(len(phi)):
                    for j_idx in range(i_idx + 2, len(phi)):
                        i = phi[i_idx]
                        j = phi[j_idx]
                        # index on the unique path list (no repeated nodes)
                        i_idx_phi = phi_node_idx[i]
                        j_idx_phi = phi_node_idx[j]
                        if d_phi[i_idx_phi, j_idx_phi] == 0:
                            # adjacency/range indicator
                            d_phi[i_idx_phi, j_idx_phi] = sum(d_phi[phi_node_idx[phi[u]], phi_node_idx[phi[u + 1]]]
                                                              for u in range(i_idx, j_idx))
                            d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]
                path_od_dists[phi_idx, od] = d_phi

                for rho_idx, rho in enumerate(deviation_paths[do]):

                    if len(path_od_dists[rho_idx, do]) == 0:
                        rho_nodes = deviation_paths_node_lists[do][rho_idx]
                        rho_node_idx = deviation_paths_node_idxs[do][rho_idx]
                        d_rho = np.zeros((len(rho_nodes), len(rho_nodes)))
                        # neighboring nodes (from edges on path), use this to get baseline distances b/w nodes on path
                        for i, j in zip(rho[:-1], rho[1:]):
                            i_idx = rho_node_idx[i]
                            j_idx = rho_node_idx[j]
                            if d_rho[i_idx, j_idx] == 0:
                                d_rho[i_idx, j_idx] = G.edges[i, j]['km']
                                d_rho[j_idx, i_idx] = d_rho[i_idx, j_idx]  # reverse
                        for i_idx in range(len(rho)):
                            for j_idx in range(i_idx + 2, len(rho)):
                                i = rho[i_idx]
                                j = rho[j_idx]
                                i_idx_rho = rho_node_idx[i]  # index on the unique path list (no repeated nodes)
                                j_idx_rho = rho_node_idx[j]
                                if d_rho[i_idx_rho, j_idx_rho] == 0:
                                    # adjacency/range indicator
                                    d_rho[i_idx_rho, j_idx_rho] = sum(d_rho[rho_node_idx[rho[u]],
                                                                            rho_node_idx[rho[u + 1]]]
                                                                      for u in range(i_idx, j_idx))
                                    d_rho[j_idx_rho, i_idx_rho] = d_rho[i_idx_rho, j_idx_rho]
                        path_od_dists[rho_idx, do] = d_rho

    A_data = []
    A_rows = []
    A_cols = []
    B_data = []
    B_rows = []
    B_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_od_comb)))
    counter = 0
    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            p = pod_comb_idx[(phi_idx, od)]

            d_phi = path_od_dists[phi_idx, od]

            print_items = False
            # if o == 'S48245004479' and d == 'S19155001645':
            #     print_items = True
            #     print(deviation_paths[od][phi_idx])
            #     print(deviation_paths_node_lists[od][phi_idx])

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                k = npod_comb_idx[(i, phi_idx, od)]
                # for z variables, states that combination p is in combination k
                B_data.append(1)
                B_rows.append(k)
                B_cols.append(p)

                counter += 1
                if counter % 10000 == 0:
                    print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_od_comb))))

                i_idx_phi = deviation_paths_node_idxs[od][phi_idx][i]  # order of i on phi without repeated nodes
                o_idx_phi = deviation_paths_node_idxs[od][phi_idx][o]
                d_idx_phi = deviation_paths_node_idxs[od][phi_idx][d]
                if print_items:
                    print('i,o,d phi::', end='\t')
                    print(i_idx_phi, o_idx_phi, d_idx_phi)
                j_ones = []

                for j_idx, j in enumerate(phi):
                    j_idx_phi = deviation_paths_node_idxs[od][phi_idx][j]
                    if j_idx_phi > i_idx_phi:
                        if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                            j_ones.append(node_idx[j])
                    elif j_idx_phi <= i_idx_phi:
                        for rho_idx, rho in enumerate(deviation_paths[do]):
                            d_rho = path_od_dists[rho_idx, do]
                            o_idx_rho = deviation_paths_node_idxs[do][rho_idx][o]
                            d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                            if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                j_ones.append(node_idx[j])
                                break
                for rho_idx, rho in enumerate(deviation_paths[do]):
                    d_rho = path_od_dists[rho_idx, do]
                    d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                    for j in rho:
                        if node_idx[j] in j_ones:
                            continue
                        j_idx_rho = deviation_paths_node_idxs[do][rho_idx][j]
                        if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                            j_ones.append(node_idx[j])

                A_data.extend([1 for _ in range(len(j_ones))])
                A_rows.extend([k for _ in range(len(j_ones))])
                A_cols.extend(j_ones)

    A_out = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_od_comb), n))
    B_out = csr_matrix((B_data, (B_rows, B_cols)), shape=(len(node_path_od_comb), len(path_od_comb)))

    A_data = []
    A_rows = []
    A_cols = []
    B_data = []
    B_rows = []
    B_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_do_comb)))
    counter = 0
    for o, d in dos_list:
        # o is actually d and d is actually o in original ods_list
        od = (o, d)  # od is actually do, or (d, o)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            p = pdo_comb_idx[(phi_idx, od)]

            d_phi = path_od_dists[phi_idx, od]

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                k = npdo_comb_idx[(i, phi_idx, od)]
                # for z variables, states that combination p is in combination k
                B_data.append(1)
                B_rows.append(k)
                B_cols.append(p)

                counter += 1
                if counter % 10000 == 0:
                    print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_do_comb))))

                i_idx_phi = deviation_paths_node_idxs[od][phi_idx][i]  # order of i on phi without repeated nodes
                o_idx_phi = deviation_paths_node_idxs[od][phi_idx][o]
                d_idx_phi = deviation_paths_node_idxs[od][phi_idx][d]

                j_ones = []
                for j_idx, j in enumerate(phi):
                    j_idx_phi = deviation_paths_node_idxs[od][phi_idx][j]
                    if j_idx_phi > i_idx_phi:
                        if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                            j_ones.append(node_idx[j])
                    elif j_idx_phi <= i_idx_phi:
                        for rho_idx, rho in enumerate(deviation_paths[do]):
                            d_rho = path_od_dists[rho_idx, do]
                            o_idx_rho = deviation_paths_node_idxs[do][rho_idx][o]
                            d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                            if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                j_ones.append(node_idx[j])
                                break
                for rho_idx, rho in enumerate(deviation_paths[do]):
                    d_rho = path_od_dists[rho_idx, do]
                    d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                    for j in rho:
                        if node_idx[j] in j_ones:
                            continue
                        j_idx_rho = deviation_paths_node_idxs[do][rho_idx][j]
                        if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                            j_ones.append(node_idx[j])

                A_data.extend([1 for _ in range(len(j_ones))])
                A_rows.extend([k for _ in range(len(j_ones))])
                A_cols.extend(j_ones)

    A_rev = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_do_comb), n))
    B_rev = csr_matrix((B_data, (B_rows, B_cols)), shape=(len(node_path_do_comb), len(path_do_comb)))

    # G.graph['debug'] = dict(
    #     path_od_dists=path_od_dists,
    #     deviation_paths=deviation_paths,
    #     deviation_paths_node_lists=deviation_paths_node_lists
    # )

    row_idxs_A_out = np.array([[i, phi_idx, o, d] for i, phi_idx, (o, d) in node_path_od_comb])
    row_idxs_A_rev = np.array([[i, phi_idx, d, o] for i, phi_idx, (d, o) in node_path_do_comb])
    row_idxs_name_type_A = np.array([['i_node_id', 'phi_idx', 'orig', 'dest'], ['str', 'int', 'str', 'str']])
    col_idxs_A = np.array(node_list),
    col_idxs_B_out = np.array([[phi_idx, o, d] for phi_idx, (o, d) in path_od_comb])
    col_idxs_B_rev = np.array([[phi_idx, d, o] for phi_idx, (d, o) in path_do_comb])
    col_idxs_name_type_B = np.array([['phi_idx', 'orig', 'dest'], ['int', 'str', 'str']])

    np.savez(mat_filepath, A_out=A_out, B_out=B_out, A_rev=A_rev, B_rev=B_rev,
             row_idxs_A_out=row_idxs_A_out, row_idxs_A_rev=row_idxs_A_rev, row_idxs_name_type_A=row_idxs_name_type_A,
             col_idxs_A=col_idxs_A, col_idxs_B_out=col_idxs_B_out, col_idxs_B_rev=col_idxs_B_rev,
             col_idxs_name_type_B=col_idxs_name_type_B)

    return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True))


def constraint_matrix_k_dc_mp(G: nx.Graph, ods: list, node_list: list, range_km: float,
                              num_shortest_paths: int = 1, od_flow_perc: float = 1):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    # if 'KCS_ex' in G.graph['railroad']:
    #     KCS_EX_PATH = '/Users/adrianhz/Library/CloudStorage/OneDrive-NorthwesternUniversity/Research/PhD Degree' \
    #                   '/Papers/Full Rollout/Manuscript/KCS Ex'
    #     mat_filepath = os.path.join(KCS_EX_PATH, G.graph['railroad'] + '_' + str(D) + '_'
    #                                 + str(od_flow_perc) + '_ifc_constraint_mat_k' + str(num_shortest_paths) + '_mp.npz')
    #     dev_filepath = os.path.join(KCS_EX_PATH, G.graph['railroad'] + '_' + str(D) + '_'
    #                                 + str(od_flow_perc) + '_ifc_constraint_dev_k' + str(num_shortest_paths) + '_mp.pkl')
    # else:
    #     mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(D) + '_'
    #                                 + str(od_flow_perc) + '_ifc_constraint_mat_k' + str(num_shortest_paths) + '_mp.npz')
    #     dev_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(D) + '_'
    #                                 + str(od_flow_perc) + '_ifc_constraint_dev_k' + str(num_shortest_paths) + '_mp.pkl')

    mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_'
                                + str(od_flow_perc) + '_ifc_constraint_mat_k' + str(num_shortest_paths) + '_mp.npz')
    dev_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_'
                                + str(od_flow_perc) + '_ifc_constraint_dev_k' + str(num_shortest_paths) + '_mp.pkl')

    if os.path.exists(mat_filepath):
        return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True)), pkl.load(open(dev_filepath, 'rb'))

    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]

    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # for k in range(len(paths)):
    #     p = paths[k]
    #     if not nx.has_path(G, source=p[0], target=p[-1]):
    #         paths.pop(k)
    #         continue
    #     p_dists = []
    #     for i, j in zip(p[:-1], p[1:]):
    #         # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
    #         p_dists.append(G.edges[i, j]['km'])
    #     infeas_idx = np.where(np.array(p_dists) > D)[0] + 1
    #
    #     if len(infeas_idx) > 0:
    #         infeas_idx = np.insert(infeas_idx, 0, 0)
    #         infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
    #         for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
    #             sub_p = p[i:j]
    #             # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
    #             if len(sub_p) > 1:
    #                 feasible_paths.append(sub_p)
    #     # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
    #     #     feasible_paths.append(p)
    #     else:
    #         feasible_paths.append(p)

    # 3. generate set of paths (incl. super paths) for each O-D pair
    # ods = [(o, d) for o, d in ods if (o, d) in od_flows.keys() and od_flows[o, d] > 0]
    # paths = {(p[0], p[-1]): p for p in paths if (p[0], p[-1]) in ods}
    # ods = [(o, d) for o, d in ods if ((o, d) in od_flows.keys() and od_flows[o, d] > 0) or
    #        ((d, o) in od_flows.keys() and od_flows[d, o] > 0)]
    paths = {(o, d): k_shortest_paths_nodes(G, source=o, target=d, k=num_shortest_paths, weight='km') for o, d in ods}
    print('ODs %d  Paths %d' % (len(ods), num_shortest_paths * len(paths)))
    deviation_paths = {od: [p for p in ps] for od, ps in paths.items()}
    # deviation_paths_node_lists = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                for p_kl in paths[k, l]:
                    # get base path for O-D pair (k, l)
                    # p_kl = paths[k, l]
                    # if i and j on base path for O-D pair (k, l)
                    if i in p_kl and j in p_kl:
                        i_idx = p_kl.index(i)
                        j_idx = p_kl.index(j)
                        # if j comes after i on this path
                        if i_idx < j_idx:
                            # orient superpath: i->k + k+1->l-1 + l->j
                            # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                            p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                            # add this new path
                            deviation_paths[i, j].append(p_iklj)
                            # deviation_paths_node_lists[i, j].append(p_kl)
                            # if the reverse O-D pair (j, i) exists in the set to consider
                            # if (j, i) in deviation_paths.keys():
                            #     # add this new path reversed
                            #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                        # if i comes after j on this path
                        else:
                            # orient superpath the opposite way: i->l + l->k + k->j
                            p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                            # add this new path
                            deviation_paths[i, j].append(p_ilkj)
                            # deviation_paths_node_lists[i, j].append(list(reversed(p_kl)))
                            # if the reverse O-D pair (j, i) exists in the set to consider
                            # if (j, i) in deviation_paths.keys():
                            #     # add this new path reversed
                            #     deviation_paths[j, i].append(list(reversed(p_ilkj)))

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    dev_idxs = dict()  # NEW
    for i, j in unique_ods:
        deviation_paths[i, j] = [list(q) for q in set(tuple(p) for p in deviation_paths[i, j])]
        deviation_paths[j, i] = [list(reversed(p)) for p in deviation_paths[i, j]]
        dev_idxs[i, j] = [set() for _ in range(len(deviation_paths[i, j]))]  # NEW

    print(len(deviation_paths))
    print(sum(len(deviation_paths[od]) for od in deviation_paths.keys()))

    # NEW
    for i, j in unique_ods:
        for p_idx, p in enumerate(deviation_paths[i, j]):
            for q_idx, q in enumerate(deviation_paths[j, i]):
                # p is complete subpath of q (or vice versa) -> q can be used to serve p (or vice versa)
                if set(p).issubset(set(q)):
                    dev_idxs[i, j][p_idx].add(q_idx)
                if set(q).issubset(set(p)):
                    dev_idxs[i, j][p_idx].add(q_idx)
        dev_idxs[j, i] = dev_idxs[i, j]
    # END NEW

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    # O-D direction
    ods_list = list(unique_ods)
    node_path_od_comb = [(i, phi, od) for od in ods_list for phi in range(len(deviation_paths[od]))
                         for i in deviation_paths[od][phi]]
    npod_comb_idx = {v: [] for v in node_path_od_comb}
    for i, v in enumerate(node_path_od_comb):
        npod_comb_idx[v].append(i)
    # npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    path_od_comb = [(p, od) for od in ods_list for p in range(len(deviation_paths[od]))]
    pod_comb_idx = {v: i for i, v in enumerate(path_od_comb)}
    # D-O (reverse) direction
    dos_list = [(d, o) for o, d in ods_list]
    node_path_do_comb = [(i, phi, do) for do in dos_list for phi in range(len(deviation_paths[do]))
                         for i in deviation_paths[do][phi]]
    npdo_comb_idx = {v: [] for v in node_path_do_comb}
    for i, v in enumerate(node_path_do_comb):
        npdo_comb_idx[v].append(i)
    # npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    path_do_comb = [(p, do) for do in dos_list for p in range(len(deviation_paths[do]))]
    pdo_comb_idx = {v: i for i, v in enumerate(path_do_comb)}

    n = len(node_list)
    node_idx = {v: i for i, v in enumerate(node_list)}
    # path_od_dists = {(p, od): -1 for od in ods for p in range(len(deviation_paths[od]))}
    path_od_dists = {(p, od): [] for od, p_idxs in deviation_paths.items() for p in range(len(p_idxs))}
    deviation_paths_node_lists = {od: {p: [] for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    for o, d in deviation_paths_node_lists.keys():
        for p in deviation_paths_node_lists[o, d].keys():
            dp = np.array(deviation_paths[o, d][p])
            # can have repeat at start, end, or both
            o_idx = np.where(dp == o)[0]
            d_idx = np.where(dp == d)[0]
            if len(o_idx) == 1 and len(d_idx) == 1:
                deviation_paths_node_lists[o, d][p] = dp.tolist()
            elif len(o_idx) > 1 and len(d_idx) == 1:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[dp_o_idx:].tolist()
            elif len(o_idx) == 1 and len(d_idx) > 1:
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[:-dp_d_idx].tolist()
            else:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][p] = dp[dp_o_idx:-dp_d_idx].tolist()

    # deviation_paths_node_lists = {od: {p: deviation_paths[od][p][:len(set(deviation_paths[od][p]))]
    #                                    for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    # deviation_paths_node_lists = {od: {p: list(set(deviation_paths[od][p]))
    #                                    for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}
    deviation_paths_node_idxs = {od: {p: {v: i for i, v in enumerate(deviation_paths_node_lists[od][p])}
                                      for p in range(len(deviation_paths[od]))} for od in deviation_paths.keys()}

    # suppose have: od = (a, b) so that do = (b, a) and have dp[od][1] = [a,1,b,2,3,4,3,2,b] (a->1->b<->2<->3<->4)
    # so that it is cleaned to dpnl[od][1] = [a,1,b,2,3,4] and
    # dp[do][1] = [b,2,3,4,3,2,b,1,a] -> dpnl[do][1] = [4,3,2,b,1,a]
    # now we have: a_idx_phi = 0, 1_idx_phi = 1, b_idx_phi = 2, ...
    # a_idx_rho = 5, 1_idx_phi = 4, b_idx_phi = 3, ...

    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            if len(path_od_dists[phi_idx, od]) == 0:
                phi_nodes = deviation_paths_node_lists[od][phi_idx]
                phi_node_idx = deviation_paths_node_idxs[od][phi_idx]
                d_phi = np.zeros((len(phi_nodes), len(phi_nodes)))
                # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
                for i, j in zip(phi[:-1], phi[1:]):
                    i_idx_phi = phi_node_idx[i]
                    j_idx_phi = phi_node_idx[j]

                    if d_phi[i_idx_phi, j_idx_phi] == 0:
                        d_phi[i_idx_phi, j_idx_phi] = G.edges[i, j]['km']
                        d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]  # reverse
                for i_idx in range(len(phi)):
                    for j_idx in range(i_idx + 2, len(phi)):
                        i = phi[i_idx]
                        j = phi[j_idx]
                        # index on the unique path list (no repeated nodes)
                        i_idx_phi = phi_node_idx[i]
                        j_idx_phi = phi_node_idx[j]
                        if d_phi[i_idx_phi, j_idx_phi] == 0:
                            # adjacency/range indicator
                            d_phi[i_idx_phi, j_idx_phi] = sum(d_phi[phi_node_idx[phi[u]], phi_node_idx[phi[u + 1]]]
                                                              for u in range(i_idx, j_idx))
                            d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]
                path_od_dists[phi_idx, od] = d_phi

                for rho_idx, rho in enumerate(deviation_paths[do]):

                    if len(path_od_dists[rho_idx, do]) == 0:
                        rho_nodes = deviation_paths_node_lists[do][rho_idx]
                        rho_node_idx = deviation_paths_node_idxs[do][rho_idx]
                        d_rho = np.zeros((len(rho_nodes), len(rho_nodes)))
                        # neighboring nodes (from edges on path), use this to get baseline distances b/w nodes on path
                        for i, j in zip(rho[:-1], rho[1:]):
                            i_idx = rho_node_idx[i]
                            j_idx = rho_node_idx[j]

                            if d_rho[i_idx, j_idx] == 0:
                                d_rho[i_idx, j_idx] = G.edges[i, j]['km']
                                d_rho[j_idx, i_idx] = d_rho[i_idx, j_idx]  # reverse
                        for i_idx in range(len(rho)):
                            for j_idx in range(i_idx + 2, len(rho)):
                                i = rho[i_idx]
                                j = rho[j_idx]
                                i_idx_rho = rho_node_idx[i]  # index on the unique path list (no repeated nodes)
                                j_idx_rho = rho_node_idx[j]
                                if d_rho[i_idx_rho, j_idx_rho] == 0:
                                    # adjacency/range indicator
                                    d_rho[i_idx_rho, j_idx_rho] = sum(d_rho[rho_node_idx[rho[u]],
                                                                            rho_node_idx[rho[u + 1]]]
                                                                      for u in range(i_idx, j_idx))
                                    d_rho[j_idx_rho, i_idx_rho] = d_rho[i_idx_rho, j_idx_rho]
                        path_od_dists[rho_idx, do] = d_rho

    A_data = []
    A_rows = []
    A_cols = []
    B_data = []
    B_rows = []
    B_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_od_comb)))
    counter = 0
    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            p = pod_comb_idx[(phi_idx, od)]

            d_phi = path_od_dists[phi_idx, od]

            print_items = False
            # if o == 'S48245004479' and d == 'S19155001645':
            #     print_items = True
            #     print(deviation_paths[od][phi_idx])
            #     print(deviation_paths_node_lists[od][phi_idx])

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                ks = npod_comb_idx[(i, phi_idx, od)]
                # for z variables, states that combination p is in combination k
                for k in ks:
                    B_data.append(1)
                    B_rows.append(k)
                    B_cols.append(p)

                counter += 1
                if counter % 10000 == 0:
                    print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_od_comb))))

                i_idx_phi = deviation_paths_node_idxs[od][phi_idx][i]  # order of i on phi without repeated nodes
                o_idx_phi = deviation_paths_node_idxs[od][phi_idx][o]
                d_idx_phi = deviation_paths_node_idxs[od][phi_idx][d]
                if print_items:
                    print('i,o,d phi::', end='\t')
                    print(i_idx_phi, o_idx_phi, d_idx_phi)
                j_ones = []

                for j_idx, j in enumerate(phi):
                    j_idx_phi = deviation_paths_node_idxs[od][phi_idx][j]
                    if j_idx_phi > i_idx_phi:
                        if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                            j_ones.append(node_idx[j])
                    elif j_idx_phi <= i_idx_phi:
                        for rho_idx, rho in enumerate(deviation_paths[do]):
                            # NEW - rho not a valid return path for phi
                            if rho_idx in dev_idxs[o, d][phi_idx]:
                                # END NEW
                                d_rho = path_od_dists[rho_idx, do]
                                o_idx_rho = deviation_paths_node_idxs[do][rho_idx][o]
                                d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                                if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                    d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                    j_ones.append(node_idx[j])
                                    break
                for rho_idx, rho in enumerate(deviation_paths[do]):
                    # NEW - rho not a valid return path for phi
                    if rho_idx in dev_idxs[o, d][phi_idx]:
                        # END NEW

                        d_rho = path_od_dists[rho_idx, do]
                        d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                        for j in rho:
                            if node_idx[j] in j_ones:
                                continue
                            j_idx_rho = deviation_paths_node_idxs[do][rho_idx][j]
                            if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                                j_ones.append(node_idx[j])

                for k in ks:
                    A_data.extend([1 for _ in range(len(j_ones))])
                    A_rows.extend([k for _ in range(len(j_ones))])
                    A_cols.extend(j_ones)

    A_out = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_od_comb), n))
    B_out = csr_matrix((B_data, (B_rows, B_cols)), shape=(len(node_path_od_comb), len(path_od_comb)))

    A_data = []
    A_rows = []
    A_cols = []
    B_data = []
    B_rows = []
    B_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_do_comb)))
    counter = 0
    for o, d in dos_list:
        # o is actually d and d is actually o in original ods_list
        od = (o, d)  # od is actually do, or (d, o)
        do = (d, o)
        for phi_idx, phi in enumerate(deviation_paths[od]):

            p = pdo_comb_idx[(phi_idx, od)]

            d_phi = path_od_dists[phi_idx, od]

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                ks = npdo_comb_idx[(i, phi_idx, od)]
                # for z variables, states that combination p is in combination k
                for k in ks:
                    B_data.append(1)
                    B_rows.append(k)
                    B_cols.append(p)

                counter += 1
                if counter % 10000 == 0:
                    print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_do_comb))))

                i_idx_phi = deviation_paths_node_idxs[od][phi_idx][i]  # order of i on phi without repeated nodes
                o_idx_phi = deviation_paths_node_idxs[od][phi_idx][o]
                d_idx_phi = deviation_paths_node_idxs[od][phi_idx][d]

                j_ones = []
                for j_idx, j in enumerate(phi):
                    j_idx_phi = deviation_paths_node_idxs[od][phi_idx][j]
                    if j_idx_phi > i_idx_phi:
                        if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                            j_ones.append(node_idx[j])
                    elif j_idx_phi <= i_idx_phi:
                        for rho_idx, rho in enumerate(deviation_paths[do]):
                            # NEW - rho not a valid return path for phi
                            if rho_idx in dev_idxs[o, d][phi_idx]:
                                # END NEW
                                d_rho = path_od_dists[rho_idx, do]
                                o_idx_rho = deviation_paths_node_idxs[do][rho_idx][o]
                                d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                                if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                    d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                    j_ones.append(node_idx[j])
                                    break
                for rho_idx, rho in enumerate(deviation_paths[do]):
                    # NEW - rho not a valid return path for phi
                    if rho_idx in dev_idxs[o, d][phi_idx]:
                        # END NEW
                        d_rho = path_od_dists[rho_idx, do]
                        d_idx_rho = deviation_paths_node_idxs[do][rho_idx][d]
                        for j in rho:
                            if node_idx[j] in j_ones:
                                continue
                            j_idx_rho = deviation_paths_node_idxs[do][rho_idx][j]
                            if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                                j_ones.append(node_idx[j])

                for k in ks:
                    A_data.extend([1 for _ in range(len(j_ones))])
                    A_rows.extend([k for _ in range(len(j_ones))])
                    A_cols.extend(j_ones)

    A_rev = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_do_comb), n))
    B_rev = csr_matrix((B_data, (B_rows, B_cols)), shape=(len(node_path_do_comb), len(path_do_comb)))

    row_idxs_A_out = np.array([[i, phi_idx, o, d] for i, phi_idx, (o, d) in node_path_od_comb])
    row_idxs_A_rev = np.array([[i, phi_idx, d, o] for i, phi_idx, (d, o) in node_path_do_comb])
    row_idxs_name_type_A = np.array([['i_node_id', 'phi_idx', 'orig', 'dest'], ['str', 'int', 'str', 'str']])
    col_idxs_A = np.array(node_list),
    col_idxs_B_out = np.array([[phi_idx, o, d] for phi_idx, (o, d) in path_od_comb])
    col_idxs_B_rev = np.array([[phi_idx, d, o] for phi_idx, (d, o) in path_do_comb])
    col_idxs_name_type_B = np.array([['phi_idx', 'orig', 'dest'], ['int', 'str', 'str']])

    np.savez(mat_filepath, A_out=A_out, B_out=B_out, A_rev=A_rev, B_rev=B_rev,
             row_idxs_A_out=row_idxs_A_out, row_idxs_A_rev=row_idxs_A_rev, row_idxs_name_type_A=row_idxs_name_type_A,
             col_idxs_A=col_idxs_A, col_idxs_B_out=col_idxs_B_out, col_idxs_B_rev=col_idxs_B_rev,
             col_idxs_name_type_B=col_idxs_name_type_B)

    pkl.dump(deviation_paths, open(dev_filepath, 'wb'))

    return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True)), deviation_paths


def comb_constraint_matrix_k_dc_mp(G: nx.Graph, ods: list, node_list: list, range_km: float,
                                   num_shortest_paths: int = 1, od_flow_perc: float = 1):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_'
                                + str(od_flow_perc) + 'comb_ifc_constraint_mat_k' + str(num_shortest_paths) + '_mp.npz')
    dev_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(range_km) + '_'
                                + str(od_flow_perc) + 'comb_ifc_constraint_dev_k' + str(num_shortest_paths) + '_mp.pkl')

    if os.path.exists(mat_filepath):
        return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True)), pkl.load(open(dev_filepath, 'rb'))

    paths = {(o, d): k_shortest_paths_nodes(G, source=o, target=d, k=num_shortest_paths, weight='km') for o, d in ods}
    print('ODs %d  Paths %d' % (len(ods), num_shortest_paths * len(paths)))
    deviation_paths = {od: [[p] for p in ps] for od, ps in paths.items()}
    # deviation_paths_node_lists = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                for p_kl in paths[k, l]:
                    # get base path for O-D pair (k, l)
                    # p_kl = paths[k, l]
                    for p_ij_idx, p_ij in enumerate(paths[i, j]):
                        # if i and j on base path for O-D pair (k, l) and p_ij is a sub-path of p_kl
                        if i in p_kl and j in p_kl and set(p_ij).issubset(set(p_kl)):
                            i_idx = p_kl.index(i)
                            j_idx = p_kl.index(j)
                            # if j comes after i on this path
                            if i_idx < j_idx:
                                # orient superpath: i->k + k+1->l-1 + l->j
                                # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                                p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                                # add this new path
                                deviation_paths[i, j][p_ij_idx].append(p_iklj)
                                # deviation_paths_node_lists[i, j].append(p_kl)
                                # if the reverse O-D pair (j, i) exists in the set to consider
                                # if (j, i) in deviation_paths.keys():
                                #     # add this new path reversed
                                #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                            # if i comes after j on this path
                            else:
                                # orient superpath the opposite way: i->l + l->k + k->j
                                p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                                # add this new path
                                deviation_paths[i, j][p_ij_idx].append(p_ilkj)

    # deviation_paths contains a collection of sets containing paths that can be grouped together
    #  - for an OD pair (i, j), deviation_paths[i, j] = [set_1, set_2, ..., set_k];
    #  - where set_s = [path_1_s, path_2_s, ..., path_p_s];
    #  - where path_1_s is a list of nodes on base path, path_2_s through path_p_s are super paths for the base path

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    # dev_idxs = dict()   # NEW
    for i, j in unique_ods:
        deviation_paths[i, j] = [[list(q) for q in set(tuple(p) for p in paths)] for paths in deviation_paths[i, j]]
        deviation_paths[j, i] = [[list(reversed(p)) for p in paths] for paths in deviation_paths[i, j]]
        # dev_idxs[i, j] = [set() for _ in range(len(deviation_paths[i, j]))]    # NEW

    print(len(deviation_paths))
    print(sum(len(deviation_paths[od][paths_idxs][p_idx]) for od, path_sets in deviation_paths.items()
              for paths_idxs, paths in enumerate(path_sets) for p_idx, _ in enumerate(paths)))

    # # NEW
    # for i, j in unique_ods:
    #     for p_idx, p in enumerate(deviation_paths[i, j]):
    #         for q_idx, q in enumerate(deviation_paths[j, i]):
    #             # p is complete subpath of q (or vice versa) -> q can be used to serve p (or vice versa)
    #             if set(p).issubset(set(q)):
    #                 dev_idxs[i, j][p_idx].add(q_idx)
    #             if set(q).issubset(set(p)):
    #                 dev_idxs[i, j][p_idx].add(q_idx)
    #     dev_idxs[j, i] = dev_idxs[i, j]
    # # END NEW

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    phi_list = {od: [(paths_idx, p_idx) for paths_idx, paths in enumerate(deviation_paths[od])
                     for p_idx, _ in enumerate(paths)] for od in deviation_paths.keys()}
    phi_list_mapper = {od: {v: i for i, v in enumerate(phi_list[od])} for od in deviation_paths.keys()}
    # O-D direction
    ods_list = list(unique_ods)
    node_path_od_comb = [(i, p_idx, rev_1, od) for od in ods_list
                         for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od])
                         for i in deviation_paths[od][phi_1_idx][phi_2_idx]
                         for rev_1, _ in enumerate(deviation_paths[od])]
    # TODO: NEW
    npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    # TODO: END NEW
    # TODO: COMMENTED
    #  why would we have more than 1 repeated npod, should be unique
    # npod_comb_idx = {v: [] for v in node_path_od_comb}
    # for i, v in enumerate(node_path_od_comb):
    #     npod_comb_idx[v].append(i)
    # TODO: END COMMENTED

    # npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    path_od_comb = [(p, od) for od in ods_list for p, _ in enumerate(phi_list[od])]
    pod_comb_idx = {v: i for i, v in enumerate(path_od_comb)}
    # D-O (reverse) direction
    dos_list = [(d, o) for o, d in ods_list]
    node_path_do_comb = [(i, p_idx, rev_1, do) for do in dos_list
                         for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[do])
                         for i in deviation_paths[do][phi_1_idx][phi_2_idx]
                         for rev_1, _ in enumerate(deviation_paths[do])]
    # TODO: NEW
    npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    # TODO: END NEW
    # TODO: COMMENTED
    #  why would we have more than 1 repeated npod, should be unique
    # npdo_comb_idx = {v: [] for v in node_path_do_comb}
    # for i, v in enumerate(node_path_do_comb):
    #     npdo_comb_idx[v].append(i)
    # TODO: END COMMENTED
    # npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    path_do_comb = [(p, do) for do in dos_list for p, _ in enumerate(phi_list[do])]
    pdo_comb_idx = {v: i for i, v in enumerate(path_do_comb)}

    n = len(node_list)
    node_idx = {v: i for i, v in enumerate(node_list)}
    # path_od_dists = {(p, od): -1 for od in ods for p in range(len(deviation_paths[od]))}
    path_od_dists = {(p, od): [] for od in deviation_paths.keys() for p, _ in enumerate(phi_list[od])}
    deviation_paths_node_lists = {od: {p: [] for p, _ in enumerate(phi_list[od])} for od in deviation_paths.keys()}
    for o, d in deviation_paths_node_lists.keys():
        for pod_idx in deviation_paths_node_lists[o, d].keys():
            phi_1_idx, phi_2_idx = phi_list[o, d][pod_idx]
            dp = np.array(deviation_paths[o, d][phi_1_idx][phi_2_idx])
            # can have repeat at start, end, or both
            o_idx = np.where(dp == o)[0]
            d_idx = np.where(dp == d)[0]
            if len(o_idx) == 1 and len(d_idx) == 1:
                deviation_paths_node_lists[o, d][pod_idx] = dp.tolist()
            elif len(o_idx) > 1 and len(d_idx) == 1:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[dp_o_idx:].tolist()
            elif len(o_idx) == 1 and len(d_idx) > 1:
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[:-dp_d_idx].tolist()
            else:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[dp_o_idx:-dp_d_idx].tolist()

    deviation_paths_node_idxs = {od: {p: {v: i for i, v in enumerate(deviation_paths_node_lists[od][p])}
                                      for p, _ in enumerate(phi_list[od])} for od in deviation_paths.keys()}

    # suppose have: od = (a, b) so that do = (b, a) and have dp[od][1] = [a,1,b,2,3,4,3,2,b] (a->1->b<->2<->3<->4)
    # so that it is cleaned to dpnl[od][1] = [a,1,b,2,3,4] and
    # dp[do][1] = [b,2,3,4,3,2,b,1,a] -> dpnl[do][1] = [4,3,2,b,1,a]
    # now we have: a_idx_phi = 0, 1_idx_phi = 1, b_idx_phi = 2, ...
    # a_idx_rho = 5, 1_idx_phi = 4, b_idx_phi = 3, ...

    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            if len(path_od_dists[p_idx, od]) == 0:
                phi_nodes = deviation_paths_node_lists[od][p_idx]
                phi_node_idx = deviation_paths_node_idxs[od][p_idx]
                d_phi = np.zeros((len(phi_nodes), len(phi_nodes)))
                # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
                for i, j in zip(phi[:-1], phi[1:]):
                    i_idx_phi = phi_node_idx[i]
                    j_idx_phi = phi_node_idx[j]

                    if d_phi[i_idx_phi, j_idx_phi] == 0:
                        d_phi[i_idx_phi, j_idx_phi] = G.edges[i, j]['km']
                        d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]  # reverse
                for i_idx in range(len(phi)):
                    for j_idx in range(i_idx + 2, len(phi)):
                        i = phi[i_idx]
                        j = phi[j_idx]
                        # index on the unique path list (no repeated nodes)
                        i_idx_phi = phi_node_idx[i]
                        j_idx_phi = phi_node_idx[j]
                        if d_phi[i_idx_phi, j_idx_phi] == 0:
                            # adjacency/range indicator
                            d_phi[i_idx_phi, j_idx_phi] = sum(d_phi[phi_node_idx[phi[u]], phi_node_idx[phi[u + 1]]]
                                                              for u in range(i_idx, j_idx))
                            d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]
                path_od_dists[p_idx, od] = d_phi

                for r_idx, (rho_1_idx, rho_2_idx) in enumerate(phi_list[do]):
                    rho = deviation_paths[do][rho_1_idx][rho_2_idx]

                    if len(path_od_dists[r_idx, do]) == 0:
                        rho_nodes = deviation_paths_node_lists[do][r_idx]
                        rho_node_idx = deviation_paths_node_idxs[do][r_idx]
                        d_rho = np.zeros((len(rho_nodes), len(rho_nodes)))
                        # neighboring nodes (from edges on path), use this to get baseline distances b/w nodes on path
                        for i, j in zip(rho[:-1], rho[1:]):
                            i_idx = rho_node_idx[i]
                            j_idx = rho_node_idx[j]

                            if d_rho[i_idx, j_idx] == 0:
                                d_rho[i_idx, j_idx] = G.edges[i, j]['km']
                                d_rho[j_idx, i_idx] = d_rho[i_idx, j_idx]  # reverse
                        for i_idx in range(len(rho)):
                            for j_idx in range(i_idx + 2, len(rho)):
                                i = rho[i_idx]
                                j = rho[j_idx]
                                i_idx_rho = rho_node_idx[i]  # index on the unique path list (no repeated nodes)
                                j_idx_rho = rho_node_idx[j]
                                if d_rho[i_idx_rho, j_idx_rho] == 0:
                                    # adjacency/range indicator
                                    d_rho[i_idx_rho, j_idx_rho] = sum(d_rho[rho_node_idx[rho[u]],
                                                                            rho_node_idx[rho[u + 1]]]
                                                                      for u in range(i_idx, j_idx))
                                    d_rho[j_idx_rho, i_idx_rho] = d_rho[i_idx_rho, j_idx_rho]
                        path_od_dists[r_idx, do] = d_rho

    A_data = []
    A_rows = []
    A_cols = []
    H_data = []
    H_rows = []
    H_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_od_comb)))
    counter = 0
    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            pod_idx = pod_comb_idx[p_idx, od]
            d_phi = path_od_dists[p_idx, od]
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            print_items = False

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                for rho_1_idx, rho_1_set in enumerate(deviation_paths[do]):
                    # for z variables, states that combination p is in combination k
                    k = npod_comb_idx[(i, p_idx, rho_1_idx, od)]
                    H_data.append(1)
                    H_rows.append(k)
                    H_cols.append(pod_idx)

                    counter += 1
                    if counter % 10000 == 0:
                        print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_od_comb))))

                    i_idx_phi = deviation_paths_node_idxs[od][p_idx][i]  # order of i on phi without repeated nodes
                    o_idx_phi = deviation_paths_node_idxs[od][p_idx][o]
                    d_idx_phi = deviation_paths_node_idxs[od][p_idx][d]
                    if print_items:
                        print('i,o,d phi::', end='\t')
                        print(i_idx_phi, o_idx_phi, d_idx_phi)
                    j_ones = []

                    for j_idx, j in enumerate(phi):
                        j_idx_phi = deviation_paths_node_idxs[od][p_idx][j]
                        if j_idx_phi > i_idx_phi:
                            if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                                j_ones.append(node_idx[j])
                        elif j_idx_phi <= i_idx_phi:
                            for rho_2_idx, _ in enumerate(rho_1_set):
                                r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                                d_rho = path_od_dists[r_idx, do]
                                o_idx_rho = deviation_paths_node_idxs[do][r_idx][o]
                                d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                                if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                    d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                    j_ones.append(node_idx[j])
                                    break
                    for rho_2_idx, rho in enumerate(rho_1_set):
                        r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                        d_rho = path_od_dists[r_idx, do]
                        d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                        for j in rho:
                            if node_idx[j] in j_ones:
                                continue
                            j_idx_rho = deviation_paths_node_idxs[do][r_idx][j]
                            if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                                j_ones.append(node_idx[j])

                    A_data.extend([1 for _ in range(len(j_ones))])
                    A_rows.extend([k for _ in range(len(j_ones))])
                    A_cols.extend(j_ones)

    A_out = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_od_comb), n))
    H_out = csr_matrix((H_data, (H_rows, H_cols)), shape=(len(node_path_od_comb), len(path_od_comb)))
    A_out[A_out > 0] = 1
    H_out[H_out > 0] = 1

    A_data = []
    A_rows = []
    A_cols = []
    H_data = []
    H_rows = []
    H_cols = []
    print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_do_comb)))
    counter = 0
    for o, d in dos_list:
        # o is actually d and d is actually o in original ods_list
        od = (o, d)  # od is actually do, or (d, o)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            pod_idx = pdo_comb_idx[p_idx, od]
            d_phi = path_od_dists[p_idx, od]
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                for rho_1_idx, rho_1_set in enumerate(deviation_paths[do]):
                    # for z variables, states that combination p is in combination k
                    k = npdo_comb_idx[(i, p_idx, rho_1_idx, od)]
                    H_data.append(1)
                    H_rows.append(k)
                    H_cols.append(pod_idx)

                    counter += 1
                    if counter % 10000 == 0:
                        print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_do_comb))))

                    i_idx_phi = deviation_paths_node_idxs[od][p_idx][i]  # order of i on phi without repeated nodes
                    o_idx_phi = deviation_paths_node_idxs[od][p_idx][o]
                    d_idx_phi = deviation_paths_node_idxs[od][p_idx][d]

                    j_ones = []
                    for j_idx, j in enumerate(phi):
                        j_idx_phi = deviation_paths_node_idxs[od][p_idx][j]
                        if j_idx_phi > i_idx_phi:
                            if d_phi[i_idx_phi, j_idx_phi] <= range_km:
                                j_ones.append(node_idx[j])
                        elif j_idx_phi <= i_idx_phi:
                            for rho_2_idx, _ in enumerate(rho_1_set):
                                r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                                d_rho = path_od_dists[r_idx, do]
                                o_idx_rho = deviation_paths_node_idxs[do][r_idx][o]
                                d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                                if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                    d_phi[o_idx_phi, j_idx_phi]) <= range_km:
                                    j_ones.append(node_idx[j])
                                    break
                    for rho_2_idx, rho in enumerate(rho_1_set):
                        r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                        d_rho = path_od_dists[r_idx, do]
                        d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                        for j in rho:
                            if node_idx[j] in j_ones:
                                continue
                            j_idx_rho = deviation_paths_node_idxs[do][r_idx][j]
                            if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= range_km:
                                j_ones.append(node_idx[j])

                    A_data.extend([1 for _ in range(len(j_ones))])
                    A_rows.extend([k for _ in range(len(j_ones))])
                    A_cols.extend(j_ones)

    A_rev = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_do_comb), n))
    H_rev = csr_matrix((H_data, (H_rows, H_cols)), shape=(len(node_path_do_comb), len(path_do_comb)))
    A_rev[A_rev > 0] = 1
    H_rev[H_rev > 0] = 1

    # G.graph['debug'] = dict(
    #     path_od_dists=path_od_dists,
    #     deviation_paths=deviation_paths,
    #     deviation_paths_node_lists=deviation_paths_node_lists
    # )

    row_idxs_A_out = np.array([[i, p_idx, rev_1_idx, o, d] for i, p_idx, rev_1_idx, (o, d) in node_path_od_comb])
    row_idxs_A_rev = np.array([[i, p_idx, rev_1_idx, d, o] for i, p_idx, rev_1_idx, (d, o) in node_path_do_comb])
    row_idxs_name_type_A = np.array([['i_node_id', 'p_idx', 'rev_1_idx', 'orig', 'dest'],
                                     ['str', 'int', 'int', 'str', 'str']])
    col_idxs_A = np.array(node_list)
    col_idxs_H_out = np.array([[p_idx, o, d] for p_idx, (o, d) in path_od_comb])
    col_idxs_H_rev = np.array([[p_idx, d, o] for p_idx, (d, o) in path_do_comb])
    col_idxs_name_type_H = np.array([['p_idx', 'orig', 'dest'], ['int', 'str', 'str']])

    np.savez(mat_filepath, A_out=A_out, H_out=H_out, A_rev=A_rev, H_rev=H_rev,
             row_idxs_A_out=row_idxs_A_out, row_idxs_A_rev=row_idxs_A_rev, row_idxs_name_type_A=row_idxs_name_type_A,
             col_idxs_A=col_idxs_A, col_idxs_H_out=col_idxs_H_out, col_idxs_H_rev=col_idxs_H_rev,
             col_idxs_name_type_H=col_idxs_name_type_H)

    pkl.dump(deviation_paths, open(dev_filepath, 'wb'))

    return convert_np_arrays_dc_mp(np.load(mat_filepath, allow_pickle=True)), deviation_paths


def comb_constraint_matrix_k_time_dc_mp(G: nx.Graph, ods: list, node_list: list, range_km_set: set,
                                        num_shortest_paths: int = 1, od_flow_perc: float = 1):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    mat_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(sorted(list(range_km_set))) + '_'
                                + str(od_flow_perc) + 'comb_ifc_constraint_mat_k' + str(num_shortest_paths) + '_mp.npz')
    dev_filepath = os.path.join(MAT_DIR, G.graph['railroad'] + '_' + str(sorted(list(range_km_set))) + '_'
                                + str(od_flow_perc) + 'comb_ifc_constraint_dev_k' + str(num_shortest_paths) + '_mp.pkl')

    if os.path.exists(mat_filepath):
        return (convert_np_arrays_time_dc_mp(np.load(mat_filepath, allow_pickle=True), range_km=list(range_km_set)),
                pkl.load(open(dev_filepath, 'rb')))

    paths = {(o, d): k_shortest_paths_nodes(G, source=o, target=d, k=num_shortest_paths, weight='km') for o, d in ods}
    # print('ODs %d  Paths %d' % (len(ods), num_shortest_paths * len(paths)))
    deviation_paths = {od: [[p] for p in ps] for od, ps in paths.items()}
    # deviation_paths_node_lists = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                for p_kl in paths[k, l]:
                    # get base path for O-D pair (k, l)
                    # p_kl = paths[k, l]
                    for p_ij_idx, p_ij in enumerate(paths[i, j]):
                        # if i and j on base path for O-D pair (k, l) and p_ij is a sub-path of p_kl
                        if i in p_kl and j in p_kl and set(p_ij).issubset(set(p_kl)):
                            i_idx = p_kl.index(i)
                            j_idx = p_kl.index(j)
                            # if j comes after i on this path
                            if i_idx < j_idx:
                                # orient superpath: i->k + k+1->l-1 + l->j
                                # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                                p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                                # add this new path
                                deviation_paths[i, j][p_ij_idx].append(p_iklj)
                                # deviation_paths_node_lists[i, j].append(p_kl)
                                # if the reverse O-D pair (j, i) exists in the set to consider
                                # if (j, i) in deviation_paths.keys():
                                #     # add this new path reversed
                                #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                            # if i comes after j on this path
                            else:
                                # orient superpath the opposite way: i->l + l->k + k->j
                                p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                                # add this new path
                                deviation_paths[i, j][p_ij_idx].append(p_ilkj)

    # deviation_paths contains a collection of sets containing paths that can be grouped together
    #  - for an OD pair (i, j), deviation_paths[i, j] = [set_1, set_2, ..., set_k];
    #  - where set_s = [path_1_s, path_2_s, ..., path_p_s];
    #  - where path_1_s is a list of nodes on base path, path_2_s through path_p_s are super paths for the base path

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    # dev_idxs = dict()   # NEW
    for i, j in unique_ods:
        deviation_paths[i, j] = [[list(q) for q in set(tuple(p) for p in paths)] for paths in deviation_paths[i, j]]
        deviation_paths[j, i] = [[list(reversed(p)) for p in paths] for paths in deviation_paths[i, j]]
        # dev_idxs[i, j] = [set() for _ in range(len(deviation_paths[i, j]))]    # NEW

    # print(len(deviation_paths))
    # print(sum(len(deviation_paths[od][paths_idxs][p_idx]) for od, path_sets in deviation_paths.items()
    #           for paths_idxs, paths in enumerate(path_sets) for p_idx, _ in enumerate(paths)))

    # # NEW
    # for i, j in unique_ods:
    #     for p_idx, p in enumerate(deviation_paths[i, j]):
    #         for q_idx, q in enumerate(deviation_paths[j, i]):
    #             # p is complete subpath of q (or vice versa) -> q can be used to serve p (or vice versa)
    #             if set(p).issubset(set(q)):
    #                 dev_idxs[i, j][p_idx].add(q_idx)
    #             if set(q).issubset(set(p)):
    #                 dev_idxs[i, j][p_idx].add(q_idx)
    #     dev_idxs[j, i] = dev_idxs[i, j]
    # # END NEW

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    phi_list = {od: [(paths_idx, p_idx) for paths_idx, paths in enumerate(deviation_paths[od])
                     for p_idx, _ in enumerate(paths)] for od in deviation_paths.keys()}
    phi_list_mapper = {od: {v: i for i, v in enumerate(phi_list[od])} for od in deviation_paths.keys()}
    # O-D direction
    ods_list = list(unique_ods)
    node_path_od_comb = [(i, p_idx, rev_1, od) for od in ods_list
                         for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od])
                         for i in deviation_paths[od][phi_1_idx][phi_2_idx]
                         for rev_1, _ in enumerate(deviation_paths[od])]
    # TODO: NEW
    npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    # TODO: END NEW
    # TODO: COMMENTED
    #  why would we have more than 1 repeated npod, should be unique
    # npod_comb_idx = {v: [] for v in node_path_od_comb}
    # for i, v in enumerate(node_path_od_comb):
    #     npod_comb_idx[v].append(i)
    # TODO: END COMMENTED

    # npod_comb_idx = {v: i for i, v in enumerate(node_path_od_comb)}
    path_od_comb = [(p, od) for od in ods_list for p, _ in enumerate(phi_list[od])]
    pod_comb_idx = {v: i for i, v in enumerate(path_od_comb)}
    # D-O (reverse) direction
    dos_list = [(d, o) for o, d in ods_list]
    node_path_do_comb = [(i, p_idx, rev_1, do) for do in dos_list
                         for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[do])
                         for i in deviation_paths[do][phi_1_idx][phi_2_idx]
                         for rev_1, _ in enumerate(deviation_paths[do])]
    # TODO: NEW
    npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    # TODO: END NEW
    # TODO: COMMENTED
    #  why would we have more than 1 repeated npod, should be unique
    # npdo_comb_idx = {v: [] for v in node_path_do_comb}
    # for i, v in enumerate(node_path_do_comb):
    #     npdo_comb_idx[v].append(i)
    # TODO: END COMMENTED
    # npdo_comb_idx = {v: i for i, v in enumerate(node_path_do_comb)}
    path_do_comb = [(p, do) for do in dos_list for p, _ in enumerate(phi_list[do])]
    pdo_comb_idx = {v: i for i, v in enumerate(path_do_comb)}

    n = len(node_list)
    node_idx = {v: i for i, v in enumerate(node_list)}
    # path_od_dists = {(p, od): -1 for od in ods for p in range(len(deviation_paths[od]))}
    path_od_dists = {(p, od): [] for od in deviation_paths.keys() for p, _ in enumerate(phi_list[od])}
    deviation_paths_node_lists = {od: {p: [] for p, _ in enumerate(phi_list[od])} for od in deviation_paths.keys()}
    for o, d in deviation_paths_node_lists.keys():
        for pod_idx in deviation_paths_node_lists[o, d].keys():
            phi_1_idx, phi_2_idx = phi_list[o, d][pod_idx]
            dp = np.array(deviation_paths[o, d][phi_1_idx][phi_2_idx])
            # can have repeat at start, end, or both
            o_idx = np.where(dp == o)[0]
            d_idx = np.where(dp == d)[0]
            if len(o_idx) == 1 and len(d_idx) == 1:
                deviation_paths_node_lists[o, d][pod_idx] = dp.tolist()
            elif len(o_idx) > 1 and len(d_idx) == 1:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[dp_o_idx:].tolist()
            elif len(o_idx) == 1 and len(d_idx) > 1:
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[:-dp_d_idx].tolist()
            else:
                dp_o_idx = (o_idx[1] - o_idx[0]) // 2
                dp_d_idx = (d_idx[1] - d_idx[0]) // 2
                deviation_paths_node_lists[o, d][pod_idx] = dp[dp_o_idx:-dp_d_idx].tolist()

    deviation_paths_node_idxs = {od: {p: {v: i for i, v in enumerate(deviation_paths_node_lists[od][p])}
                                      for p, _ in enumerate(phi_list[od])} for od in deviation_paths.keys()}

    # suppose have: od = (a, b) so that do = (b, a) and have dp[od][1] = [a,1,b,2,3,4,3,2,b] (a->1->b<->2<->3<->4)
    # so that it is cleaned to dpnl[od][1] = [a,1,b,2,3,4] and
    # dp[do][1] = [b,2,3,4,3,2,b,1,a] -> dpnl[do][1] = [4,3,2,b,1,a]
    # now we have: a_idx_phi = 0, 1_idx_phi = 1, b_idx_phi = 2, ...
    # a_idx_rho = 5, 1_idx_phi = 4, b_idx_phi = 3, ...

    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            if len(path_od_dists[p_idx, od]) == 0:
                phi_nodes = deviation_paths_node_lists[od][p_idx]
                phi_node_idx = deviation_paths_node_idxs[od][p_idx]
                d_phi = np.zeros((len(phi_nodes), len(phi_nodes)))
                # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
                for i, j in zip(phi[:-1], phi[1:]):
                    i_idx_phi = phi_node_idx[i]
                    j_idx_phi = phi_node_idx[j]

                    if d_phi[i_idx_phi, j_idx_phi] == 0:
                        d_phi[i_idx_phi, j_idx_phi] = G.edges[i, j]['km']
                        d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]  # reverse
                for i_idx in range(len(phi)):
                    for j_idx in range(i_idx + 2, len(phi)):
                        i = phi[i_idx]
                        j = phi[j_idx]
                        # index on the unique path list (no repeated nodes)
                        i_idx_phi = phi_node_idx[i]
                        j_idx_phi = phi_node_idx[j]
                        if d_phi[i_idx_phi, j_idx_phi] == 0:
                            # adjacency/range indicator
                            d_phi[i_idx_phi, j_idx_phi] = sum(d_phi[phi_node_idx[phi[u]], phi_node_idx[phi[u + 1]]]
                                                              for u in range(i_idx, j_idx))
                            d_phi[j_idx_phi, i_idx_phi] = d_phi[i_idx_phi, j_idx_phi]
                path_od_dists[p_idx, od] = d_phi

                for r_idx, (rho_1_idx, rho_2_idx) in enumerate(phi_list[do]):
                    rho = deviation_paths[do][rho_1_idx][rho_2_idx]

                    if len(path_od_dists[r_idx, do]) == 0:
                        rho_nodes = deviation_paths_node_lists[do][r_idx]
                        rho_node_idx = deviation_paths_node_idxs[do][r_idx]
                        d_rho = np.zeros((len(rho_nodes), len(rho_nodes)))
                        # neighboring nodes (from edges on path), use this to get baseline distances b/w nodes on path
                        for i, j in zip(rho[:-1], rho[1:]):
                            i_idx = rho_node_idx[i]
                            j_idx = rho_node_idx[j]

                            if d_rho[i_idx, j_idx] == 0:
                                d_rho[i_idx, j_idx] = G.edges[i, j]['km']
                                d_rho[j_idx, i_idx] = d_rho[i_idx, j_idx]  # reverse
                        for i_idx in range(len(rho)):
                            for j_idx in range(i_idx + 2, len(rho)):
                                i = rho[i_idx]
                                j = rho[j_idx]
                                i_idx_rho = rho_node_idx[i]  # index on the unique path list (no repeated nodes)
                                j_idx_rho = rho_node_idx[j]
                                if d_rho[i_idx_rho, j_idx_rho] == 0:
                                    # adjacency/range indicator
                                    d_rho[i_idx_rho, j_idx_rho] = sum(d_rho[rho_node_idx[rho[u]],
                                                                            rho_node_idx[rho[u + 1]]]
                                                                      for u in range(i_idx, j_idx))
                                    d_rho[j_idx_rho, i_idx_rho] = d_rho[i_idx_rho, j_idx_rho]
                        path_od_dists[r_idx, do] = d_rho

    A_data = {r_t: [] for r_t in range_km_set}
    A_rows = {r_t: [] for r_t in range_km_set}
    A_cols = {r_t: [] for r_t in range_km_set}
    # A_data = []
    # A_rows = []
    # A_cols = []
    H_data = []
    H_rows = []
    H_cols = []
    # print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_od_comb)))
    counter = 0
    for o, d in ods_list:
        od = (o, d)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            pod_idx = pod_comb_idx[p_idx, od]
            d_phi = path_od_dists[p_idx, od]
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            print_items = False

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                for rho_1_idx, rho_1_set in enumerate(deviation_paths[do]):
                    # for z variables, states that combination p is in combination k
                    k = npod_comb_idx[(i, p_idx, rho_1_idx, od)]
                    H_data.append(1)
                    H_rows.append(k)
                    H_cols.append(pod_idx)

                    # counter += 1
                    # if counter % 10000 == 0:
                    #     print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_od_comb))))

                    i_idx_phi = deviation_paths_node_idxs[od][p_idx][i]  # order of i on phi without repeated nodes
                    o_idx_phi = deviation_paths_node_idxs[od][p_idx][o]
                    d_idx_phi = deviation_paths_node_idxs[od][p_idx][d]
                    if print_items:
                        print('i,o,d phi::', end='\t')
                        print(i_idx_phi, o_idx_phi, d_idx_phi)

                    for r_t in range_km_set:
                        j_ones = []
                        for j_idx, j in enumerate(phi):
                            j_idx_phi = deviation_paths_node_idxs[od][p_idx][j]
                            if j_idx_phi > i_idx_phi:
                                if d_phi[i_idx_phi, j_idx_phi] <= r_t:
                                    j_ones.append(node_idx[j])
                            elif j_idx_phi <= i_idx_phi:
                                for rho_2_idx, _ in enumerate(rho_1_set):
                                    r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                                    d_rho = path_od_dists[r_idx, do]
                                    o_idx_rho = deviation_paths_node_idxs[do][r_idx][o]
                                    d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                                    if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                        d_phi[o_idx_phi, j_idx_phi]) <= r_t:
                                        j_ones.append(node_idx[j])
                                        break
                        for rho_2_idx, rho in enumerate(rho_1_set):
                            r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                            d_rho = path_od_dists[r_idx, do]
                            d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                            for j in rho:
                                if node_idx[j] in j_ones:
                                    continue
                                j_idx_rho = deviation_paths_node_idxs[do][r_idx][j]
                                if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= r_t:
                                    j_ones.append(node_idx[j])

                        A_data[r_t].extend([1 for _ in range(len(j_ones))])
                        A_rows[r_t].extend([k for _ in range(len(j_ones))])
                        A_cols[r_t].extend(j_ones)

    # A_out = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_od_comb), n))
    A_out = dict()
    for r_t in range_km_set:
        A_out[r_t] = csr_matrix((A_data[r_t], (A_rows[r_t], A_cols[r_t])), shape=(len(node_path_od_comb), n))
        A_out[r_t][A_out[r_t] > 0] = 1
    H_out = csr_matrix((H_data, (H_rows, H_cols)), shape=(len(node_path_od_comb), len(path_od_comb)))
    H_out[H_out > 0] = 1

    # A_data = []
    # A_rows = []
    # A_cols = []
    A_data = {r_t: [] for r_t in range_km_set}
    A_rows = {r_t: [] for r_t in range_km_set}
    A_cols = {r_t: [] for r_t in range_km_set}
    H_data = []
    H_rows = []
    H_cols = []
    # print('MATRIX # ROWS:: {v1}'.format(v1=len(node_path_do_comb)))
    counter = 0
    for o, d in dos_list:
        # o is actually d and d is actually o in original ods_list
        od = (o, d)  # od is actually do, or (d, o)
        do = (d, o)
        for p_idx, (phi_1_idx, phi_2_idx) in enumerate(phi_list[od]):
            pod_idx = pdo_comb_idx[p_idx, od]
            d_phi = path_od_dists[p_idx, od]
            phi = deviation_paths[od][phi_1_idx][phi_2_idx]

            for i_idx, i in enumerate(phi):
                # i_idx is the order of i on phi (which may contain repeated nodes)
                for rho_1_idx, rho_1_set in enumerate(deviation_paths[do]):
                    # for z variables, states that combination p is in combination k
                    k = npdo_comb_idx[(i, p_idx, rho_1_idx, od)]
                    H_data.append(1)
                    H_rows.append(k)
                    H_cols.append(pod_idx)

                    # counter += 1
                    # if counter % 10000 == 0:
                    #     print('STATUS:: {v1}, {v2}%'.format(v1=counter, v2=round(100 * counter / len(node_path_do_comb))))

                    i_idx_phi = deviation_paths_node_idxs[od][p_idx][i]  # order of i on phi without repeated nodes
                    o_idx_phi = deviation_paths_node_idxs[od][p_idx][o]
                    d_idx_phi = deviation_paths_node_idxs[od][p_idx][d]

                    for r_t in range_km_set:
                        j_ones = []
                        for j_idx, j in enumerate(phi):
                            j_idx_phi = deviation_paths_node_idxs[od][p_idx][j]
                            if j_idx_phi > i_idx_phi:
                                if d_phi[i_idx_phi, j_idx_phi] <= r_t:
                                    j_ones.append(node_idx[j])
                            elif j_idx_phi <= i_idx_phi:
                                for rho_2_idx, _ in enumerate(rho_1_set):
                                    r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                                    d_rho = path_od_dists[r_idx, do]
                                    o_idx_rho = deviation_paths_node_idxs[do][r_idx][o]
                                    d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                                    if (d_phi[i_idx_phi, d_idx_phi] + d_rho[o_idx_rho, d_idx_rho] +
                                        d_phi[o_idx_phi, j_idx_phi]) <= r_t:
                                        j_ones.append(node_idx[j])
                                        break
                        for rho_2_idx, rho in enumerate(rho_1_set):
                            r_idx = phi_list_mapper[do][rho_1_idx, rho_2_idx]
                            d_rho = path_od_dists[r_idx, do]
                            d_idx_rho = deviation_paths_node_idxs[do][r_idx][d]
                            for j in rho:
                                if node_idx[j] in j_ones:
                                    continue
                                j_idx_rho = deviation_paths_node_idxs[do][r_idx][j]
                                if d_phi[i_idx_phi, d_idx_phi] + d_rho[d_idx_rho, j_idx_rho] <= r_t:
                                    j_ones.append(node_idx[j])

                        A_data[r_t].extend([1 for _ in range(len(j_ones))])
                        A_rows[r_t].extend([k for _ in range(len(j_ones))])
                        A_cols[r_t].extend(j_ones)

    # A_rev = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(node_path_do_comb), n))
    A_rev = dict()
    for r_t in range_km_set:
        A_rev[r_t] = csr_matrix((A_data[r_t], (A_rows[r_t], A_cols[r_t])), shape=(len(node_path_do_comb), n))
        A_rev[r_t][A_rev[r_t] > 0] = 1
    H_rev = csr_matrix((H_data, (H_rows, H_cols)), shape=(len(node_path_do_comb), len(path_do_comb)))
    H_rev[H_rev > 0] = 1

    # G.graph['debug'] = dict(
    #     path_od_dists=path_od_dists,
    #     deviation_paths=deviation_paths,
    #     deviation_paths_node_lists=deviation_paths_node_lists
    # )

    row_idxs_A_out = np.array([[i, p_idx, rev_1_idx, o, d] for i, p_idx, rev_1_idx, (o, d) in node_path_od_comb])
    row_idxs_A_rev = np.array([[i, p_idx, rev_1_idx, d, o] for i, p_idx, rev_1_idx, (d, o) in node_path_do_comb])
    row_idxs_name_type_A = np.array([['i_node_id', 'p_idx', 'rev_1_idx', 'orig', 'dest'],
                                     ['str', 'int', 'int', 'str', 'str']])
    col_idxs_A = np.array(node_list)
    col_idxs_H_out = np.array([[p_idx, o, d] for p_idx, (o, d) in path_od_comb])
    col_idxs_H_rev = np.array([[p_idx, d, o] for p_idx, (d, o) in path_do_comb])
    col_idxs_name_type_H = np.array([['p_idx', 'orig', 'dest'], ['int', 'str', 'str']])

    A_out_args = {f'A_out_{r_t}': A_out[r_t] for r_t in range_km_set}
    A_rev_args = {f'A_rev_{r_t}': A_rev[r_t] for r_t in range_km_set}
    all_args = {'H_out': H_out, 'H_rev': H_rev, 'row_idxs_A_out': row_idxs_A_out, 'row_idxs_A_rev': row_idxs_A_rev,
                'row_idxs_name_type_A': row_idxs_name_type_A, 'col_idxs_A': col_idxs_A,
                'col_idxs_H_out': col_idxs_H_out, 'col_idxs_H_rev': col_idxs_H_rev,
                'col_idxs_name_type_H': col_idxs_name_type_H}
    all_args.update(A_out_args)
    all_args.update(A_rev_args)
    np.savez(mat_filepath, **all_args)
    # np.savez(mat_filepath, A_out=A_out, H_out=H_out, A_rev=A_rev, H_rev=H_rev,
    #          row_idxs_A_out=row_idxs_A_out, row_idxs_A_rev=row_idxs_A_rev, row_idxs_name_type_A=row_idxs_name_type_A,
    #          col_idxs_A=col_idxs_A, col_idxs_H_out=col_idxs_H_out, col_idxs_H_rev=col_idxs_H_rev,
    #          col_idxs_name_type_H=col_idxs_name_type_H)

    pkl.dump(deviation_paths, open(dev_filepath, 'wb'))

    return (convert_np_arrays_time_dc_mp(np.load(mat_filepath, allow_pickle=True), range_km=list(range_km_set)),
            deviation_paths)