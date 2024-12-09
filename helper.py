from util import *


# INPUT


def load_dict_from_json(filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'r') as fr:
        return json.load(fr)


def load_splc_station_master():

    filename = 'SPLC_station_master.csv'
    filepath = os.path.join(PARAM_DIR, filename)
    df_n = pd.read_csv(filepath, header=0)
    df_n['SPLC'] = df_n['SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))

    return df_n


def load_us_cities():

    filename = 'uscities.csv'
    filepath = os.path.join(PARAM_DIR, filename)

    return pd.read_csv(filepath, header=0)


# OUTPUT


def write_dict_to_json(d: dict, filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'w') as fw:
        json.dump(d, fw)


# GENERAL


def extract_rr(df: pd.DataFrame, rr: str, forecast=False):
    # extract rows with data for rr from df

    if forecast:
        agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
                   'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
                   'Expanded Car-Miles': np.sum,
                   'Expanded Container-Miles': np.sum, 'Total Distance Mean': np.mean,
                   'Total Distance Standard Deviation': np.std, 'Total Distance Max': np.max,
                   'Total Distance Min': np.min,
                   '2020 Expanded Tons': np.sum, '2020 Expanded Ton-Miles': np.sum,
                   '2023 Expanded Tons': np.sum, '2023 Expanded Ton-Miles': np.sum,
                   '2025 Expanded Tons': np.sum, '2025 Expanded Ton-Miles': np.sum,
                   '2030 Expanded Tons': np.sum, '2030 Expanded Ton-Miles': np.sum,
                   '2035 Expanded Tons': np.sum, '2035 Expanded Ton-Miles': np.sum,
                   '2040 Expanded Tons': np.sum, '2040 Expanded Ton-Miles': np.sum,
                   '2045 Expanded Tons': np.sum, '2045 Expanded Ton-Miles': np.sum,
                   '2050 Expanded Tons': np.sum, '2050 Expanded Ton-Miles': np.sum}
    else:
        agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
                   'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
                   'Expanded Car-Miles': np.sum,
                   'Expanded Container-Miles': np.sum, 'Total Distance Mean': np.mean,
                   'Total Distance Standard Deviation': np.std, 'Total Distance Max': np.max,
                   'Total Distance Min': np.min}
    # # apply aggregation function
    # return df_class1.agg(agg_fxn)

    if rr != 'WCAN' and rr != 'EAST' and rr != 'USA1':
        return df.loc[rr]

    if rr == 'WCAN':
        rrs = ['BNSF', 'CP', 'CN']
    elif rr == 'EAST':
        rrs = ['CSXT', 'NS', 'KCS']
    elif rr == 'USA1':
        rrs = ['BNSF', 'NS', 'KCS']

    orig_idx_names = list(df.index.names)
    orig_idx_names.remove('Railroad')
    # df.reset_index(level=list(set(orig_idx_names).difference({'Railroad'})), inplace=True)
    df.rename(index={r: rr for r in rrs}, level='Railroad', inplace=True)
    df = df.loc[rr]
    # df.groupby(by=orig_idx_names).agg(agg_fxn)
    # apply aggregation function
    return df.groupby(by=orig_idx_names).agg(agg_fxn)


def gurobi_suppress_output(suppress_output=True):

    env = gp.Env(empty=True)
    if suppress_output:
        env.setParam('OutputFlag', 0)
    env.start()
    return env


# ROUTING ALGORITHMS


def shortest_path_edges(G: nx.DiGraph, source, target, weight='km', inter_nodes: list = []) -> list:
    """
    Return list of edges on shortest path in G
    between source and target that passes through inter_nodes in the order listed
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param weight:
    :param inter_nodes:
    :return: [list] of edges, e.g., [(1, 2), (2, 3)] for path from 1->3

    Cicero: 17031000719, LA: 6037003164
    [Galesburg, Burlington, Barstow]: [17095001986, 19057001911, 6071002447]
    s = 17031000719
    e = 6037003164
    t = [17095001986, 19057001911, 6071002447]
    """

    if not nx.has_path(G, source=source, target=target):
        # if path does not exist, return empty list
        return []

    s = source
    node_path = [s]
    for v in inter_nodes:
        if not nx.has_path(G, source=s, target=v):
            # if path does not exist, return empty list
            return []
        node_path.extend(nx.shortest_path(G, s, v, weight=weight)[1:])
        s = v

    node_path.extend(nx.shortest_path(G, s, target, weight=weight)[1:])

    return node_to_edge_path(node_path)


def node_to_edge_path(node_path: list):
    # returns edge format of <node_path> e.g., if <node_path> = [0, 1, 2, 3], returns [(0, 1), (1, 2), (2, 3)]

    return list(zip(node_path[:-1], node_path[1:]))


def shortest_path_nodes(G: nx.Graph, source: int, target: int, weight='km', inter_nodes=None) -> list:
    """
    Return shortest path in G between source and target that passes through inter_nodes in the order listed
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param weight:
    :param inter_nodes:
    :return:

    Cicero: 17031000719, LA: 6037003164
    [Galesburg, Burlington, Barstow]: [17095001986, 19057001911, 6071002447]
    s = 17031000719
    e = 6037003164
    t = [17095001986, 19057001911, 6071002447]
    """

    if inter_nodes is None:
        inter_nodes = []
    # update nodeids to the correct super nodeids in G, if they are grouped as such
    inter_nodes = [updated_node_name(G, n) for n in inter_nodes]

    s = updated_node_name(G, source)
    node_path = [s]
    for v in inter_nodes:
        node_path.extend(nx.shortest_path(G, source=s, target=v, weight=weight)[1:])
        s = v

    node_path.extend(nx.shortest_path(G, source=s, target=updated_node_name(G, target), weight=weight)[1:])

    return node_path


def k_shortest_paths_nodes(G: nx.Graph, source: int, target: int, k: int, weight='km'):

    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def k_shortest_paths_edges(G: nx.Graph, source: int, target: int,  k: int, weight='km'):

    return [node_to_edge_path(p) for p in k_shortest_paths_nodes(G=G, source=source, target=target, k=k, weight=weight)]


def shortest_path_path_length(G: nx.Graph, source: int, target: int, weight='km', inter_nodes=None) -> (list, float):
    """
    Return shortest path path and length in G between source and target that passes through inter_nodes in order listed
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param weight:
    :param inter_nodes:
    :return:

    Cicero: 17031000719, LA: 6037003164
    [Galesburg, Burlington, Barstow]: [17095001986, 19057001911, 6071002447]
    s = 17031000719
    e = 6037003164
    t = [17095001986, 19057001911, 6071002447]
    """

    if inter_nodes is None:
        inter_nodes = []
    # update nodeids to the correct super nodeids in G, if they are grouped as such
    inter_nodes = [updated_node_name(G, n) for n in inter_nodes]

    s = updated_node_name(G, source)
    node_path = [s]
    for v in inter_nodes:
        node_path.extend(nx.shortest_path(G, source=s, target=v, weight=weight)[1:])
        s = v

    node_path.extend(nx.shortest_path(G, source=s, target=updated_node_name(G, target), weight=weight)[1:])

    dist_mi = 0
    for (u, v) in zip(node_path[0:-1], node_path[1:]):
        dist_mi += G.edges[u, v][weight]

    return node_path, dist_mi


# GRAPH MANAGEMENT


def updated_node_name(G: nx.Graph, node):
    # return updated node name that contains the node <node> in G

    find_node = False
    if node not in G:
        find_node = True
    for n in G:
        if find_node:
            if node in G.nodes[n]['original_nodeids']:
                return n
    if find_node:
        return None

    return node


def splc_to_node(G: nx.Graph) -> dict:
    # return dict indexed by splc codes with nodeid values; for routing
    # e.g., splc_node_dict[<splc>] = <nodeid>

    splc_node_dict = dict()

    for n in G:
        splcs = G.nodes[n]['splc']
        if not isinstance(splcs, list):
            splcs = [splcs]
        for s in splcs:
            splc_node_dict[s] = n

    return splc_node_dict


# FACILITY LOCATION


def od_pairs(G: nx.Graph, source=None, target=None, intertypes=None) -> list:
    """
    Generate list of OD pairs possible in G
    :param G: [nx.Graph] railroad name
    :param source: [None]/[int]/[list]
    :param target: [None]/[int]/[list]
    :param intertypes: [None]/[set]
    :return: [list] list of tuples with OD pairs, e.g., [(O1, D1), (O1, D2)]
    """

    if intertypes is None:
        intertypes = {'T', 'P'}

    if source is None and target is None:
        # all pairs shortest path between nodes of 'inttype' in intertypes
        nodes = [n for n in G.nodes() if G.nodes[n]['inttype'] in intertypes]
        n = len(nodes)
        return [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    if type(source) is int or type(source) is str:
        source = [source]
    if type(target) is int or type(target) is str:
        target = [target]

    if source is None:
        # target to all (target 'becomes' source, does not affect anything)
        return [(t, n) for t in target for n in G.nodes() if G.nodes[n]['inttype'] in intertypes and not t == n]
    if target is None:
        # source to all
        return [(s, n) for s in source for n in G.nodes() if G.nodes[n]['inttype'] in intertypes and not s == n]
    # many-to-many
    return [(s, t) for s in source for t in target if not s == t]


def input_cleaning(G: nx.DiGraph, time_horizon: list, od_flows: dict,
                   budgets: dict = None, flow_mins: dict = None,
                   facility_costs: dict = None, discount_rates: any = None):
    node_list = list(G.nodes())
    ods = list(od_flows.keys())

    candidate_facilities = None
    # candidate facilites not provided, assume all facilities
    if candidate_facilities is None:
        candidate_facilities = {t: set(node_list) for t in time_horizon}
    # provided candidate facilities not for full time_horizon, extend
    missing_times = [t for t in time_horizon if t not in candidate_facilities.keys()]
    if len(missing_times) > 0:
        for t in missing_times:
            candidate_facilities[t] = set(node_list)
    # ensure all candidate facilities are subsets: a candidate facility at step t is included in all future steps t+k
    for t_idx in range(len(time_horizon) - 1):
        if not candidate_facilities[time_horizon[t_idx]].issubset(candidate_facilities[time_horizon[t_idx + 1]]):
            candidate_facilities[time_horizon[t_idx + 1]].add(
                candidate_facilities[time_horizon[t_idx]].difference(candidate_facilities[time_horizon[t_idx + 1]]))

    # budgets not provided, assume to be # of facilities
    if budgets is None:
        budgets = {t: len(candidate_facilities[t]) for t in time_horizon}
    # provided budgets not for full time_horizon, extend
    else:
        missing_times = [t for t in time_horizon if t not in budgets.keys()]
        if len(missing_times) > 0:
            for t in missing_times:
                budgets[t] = len(candidate_facilities[t])

    # facility costs not provided, assume all candidate facilities have the same cost
    if facility_costs is None:
        facility_costs = {(n, t): 1 for t in time_horizon for n in node_list if n in candidate_facilities[t]}

    facility_costs = {t: np.array([facility_costs[j, t] if (j, t) in facility_costs.keys() else 1
                                   for j in node_list]).transpose() for t in time_horizon}

    # facilities with no costs specified
    # if len(set(node_list).difference(set(facility_costs.keys()))) != 0:
    #     missing_facilities = set(node_list).difference(set(facility_costs.keys()))
    #     for d in missing_facilities:
    #         facility_costs[d] = {t: np.inf for t in time_horizon}
    # facilities with missing time_horizon costs
    # missing_times = [n for n in node_list if len(facility_costs[n]) < T]
    # if len(missing_times) > 0:
    #     t0 = time_horizon[0]
    #     for d in missing_times:
    #         # if the smallest time period is empty, set it to np.inf
    #         if t0 not in facility_costs[d].keys():
    #             facility_costs[d][t0] = np.inf
    #     # fill in remaining time steps
    #     for t_idx in range(1, len(time_horizon)):
    #         for d in missing_times:
    #             # if the value for this time step is missing, take on the value of the previously available time step
    #             if time_horizon[t_idx] not in facility_costs[d].keys():
    #                 facility_costs[d][time_horizon[t_idx]] = facility_costs[d][time_horizon[t_idx - 1]]
    # facility_costs = {(n, t): facility_costs[n][t] for n, t in product(node_list, time_horizon)}

    # discount_rates not provided, assumed all to be 1
    if discount_rates is None:
        discount_rates = {t: 1 for t in time_horizon}
    # provided discount_rates not for full time_horizon, extend
    elif isinstance(discount_rates, float) or isinstance(discount_rates, int):
        discount_rates = {t: (1 / (1 + discount_rates)) ** t for t in time_horizon}
    else:
        missing_times = [t for t in time_horizon if t not in discount_rates.keys()]
        if len(missing_times) > 0:
            for t in missing_times:
                discount_rates[t] = 1

    # OD pairs with missing time_horizon flows
    missing_times = set(od for od in ods if not isinstance(od_flows[od], dict))
    if len(missing_times) > 0:
        t0 = time_horizon[0]
        for od in missing_times:
            # if the smallest time period is empty, set it to 0
            if not isinstance(od_flows[od], dict):
                od_flows[od] = {t0: od_flows[od]}
                # fill in remaining time steps
                for t_idx in range(1, len(time_horizon)):
                    # take on the value of the previously available time step
                    od_flows[od][time_horizon[t_idx]] = od_flows[od][time_horizon[t_idx - 1]]

    return od_flows, budgets, facility_costs, discount_rates


def covered_graph(G: nx.DiGraph, range_km: float, extend_graph=False):
    # facility nodes and edges
    time_horizon = G.graph['framework']['time_horizon']
    selected_facilities = G.graph['framework']['selected_facilities']
    if extend_graph:
        # includes all those edges within the range (out-and-back-in) of the selected facilities
        covered_edges = {t: set(path_edges_covered(G, fac_set=selected_facilities[t], range_km=range_km, weight='km'))
                         for t in time_horizon}
    else:
        # includes only those edges along the selected O-D paths
        covered_path_edges = G.graph['framework']['covered_path_edges']
        covered_edges = {t: set((u, v) for p in covered_path_edges[t].values() for u, v in p).union(
            set((v, u) for p in covered_path_edges[t].values() for u, v in p))
            for t in time_horizon}
    # include both endpoints of each covered edge as a covered node
    covered_nodes = {t: set(u for u, _ in covered_edges[t]).union(set(v for _, v in covered_edges[t]))
                     for t in time_horizon}

    for n in G.nodes:
        # suppose time_horizon = [0, 1, 2, 3],
        #  -then G.nodes[n]['facility'] = {0: 0, 1: 1, 2: 1, 3: 1} means a facility was placed at n in time step 1
        #  -and G.nodes[n]['covered'] = {0: 0, 1: 0, 2: 1, 3: 1} means node n was covered starting in time step 2
        G.nodes[n]['facility'] = {t: n in selected_facilities[t] for t in time_horizon}
        G.nodes[n]['covered'] = {t: n in covered_nodes[t] for t in time_horizon}

    for u, v in G.edges:
        G.edges[u, v]['covered'] = {t: (u, v) in covered_edges[t] for t in time_horizon}

    G.graph['number_facilities'] = {t: sum(G.nodes[i]['facility'][t] for i in G) for t in time_horizon}

    return G


def covered_graph_time(G: nx.DiGraph, range_km: dict, extend_graph=False):
    # facility nodes and edges
    time_horizon = G.graph['framework']['time_horizon']
    selected_facilities = G.graph['framework']['selected_facilities']
    if extend_graph:
        # includes all those edges within the range (out-and-back-in) of the selected facilities
        covered_edges = {t: set(path_edges_covered(G, fac_set=selected_facilities[t],
                                                   range_km=range_km[t], weight='km'))
                         for t in time_horizon}
    else:
        # includes only those edges along the selected O-D paths
        covered_path_edges = G.graph['framework']['covered_path_edges']
        covered_edges = {t: set((u, v) for p in covered_path_edges[t].values() for u, v in p).union(
            set((v, u) for p in covered_path_edges[t].values() for u, v in p))
            for t in time_horizon}
    # include both endpoints of each covered edge as a covered node
    covered_nodes = {t: set(u for u, _ in covered_edges[t]).union(set(v for _, v in covered_edges[t]))
                     for t in time_horizon}

    for n in G.nodes:
        # suppose time_horizon = [0, 1, 2, 3],
        #  -then G.nodes[n]['facility'] = {0: 0, 1: 1, 2: 1, 3: 1} means a facility was placed at n in time step 1
        #  -and G.nodes[n]['covered'] = {0: 0, 1: 0, 2: 1, 3: 1} means node n was covered starting in time step 2
        G.nodes[n]['facility'] = {t: n in selected_facilities[t] for t in time_horizon}
        G.nodes[n]['covered'] = {t: n in covered_nodes[t] for t in time_horizon}

    for u, v in G.edges:
        G.edges[u, v]['covered'] = {t: (u, v) in covered_edges[t] for t in time_horizon}

    G.graph['number_facilities'] = {t: sum(G.nodes[i]['facility'][t] for i in G) for t in time_horizon}

    return G

def path_edges_covered(G: nx.Graph, fac_set: set, range_km: float, weight: str = 'km') -> list:
    # create list of edges connected by selected facilities within the range

    node_list = list(G.nodes)
    dist_mat = nx.floyd_warshall_numpy(G=G, nodelist=node_list, weight=weight)

    node_idx_dict = {node_list[i]: i for i in range(len(node_list))}
    fac_idxs = [node_idx_dict[i] for i in fac_set]

    visited_edges = set()
    covered_edges = set()
    for i, j in G.edges:
        u, v = (node_idx_dict[i], node_idx_dict[j])
        if (i, j) not in visited_edges:
            visited_edges.update({(i, j), (j, i)})
            d_ku = dist_mat[fac_idxs, u].min()
            d_vk = dist_mat[v, fac_idxs].min()
            if d_ku + G.edges[i, j][weight] + d_vk <= range_km:
                covered_edges.update({(i, j), (j, i)})
    # path_edges = set()
    # for u in path_dict.keys():
    #     if u not in fac_set:
    #         continue
    #     for v in path_dict[u].keys():
    #         if v not in fac_set or u == v:
    #             continue
    #         p = path_dict[u][v]
    #         edges = {e for e in zip(p[:-1], p[1:])}
    #         path_edges.update(edges)
    #
    # path_dict_d2 = dict(nx.all_pairs_dijkstra_path(G, cutoff=D / 2, weight=weight))
    # for u in path_dict_d2.keys():
    #     if u not in fac_set:
    #         continue
    #     for v in path_dict_d2[u].keys():
    #         # print(nx.shortest_path(G, source=u, target=v))
    #         p = path_dict_d2[u][v]
    #         edges = {e for e in zip(p[:-1], p[1:])}
    #         path_edges.update(edges)

    return list(covered_edges)


def y_warm_start_random(node_list: list, time_horizon: list, budgets: dict, final_fac: list = None, seed=None):
    t0 = time.time()
    if seed is None:
        seed = datetime.now().microsecond
    # initialize random seed
    print('SEED:: {v1}'.format(v1=seed))
    np.random.seed(seed)
    # if final_fac (boundary conditions on final facilities) not restricted or provided
    if not final_fac:
        # randomly select the final facilities from all nodes
        cum_budgets = {t_step: sum(budgets[t] for t in time_horizon[:t_idx + 1])
                       for t_idx, t_step in enumerate(time_horizon)}
        final_fac_node_idxs = np.random.choice(np.array([i for i in range(len(node_list))]),
                                               cum_budgets[time_horizon[-1]], replace=False)
        final_fac = [node_list[j] for j in final_fac_node_idxs]

    # randomly select each facility subset from <final_fac> based on <budgets>
    final_fac_idxs = set(i for i in range(len(final_fac)))
    idxs_choice = set()
    y_ws = {t: [] for t in time_horizon}
    for t in time_horizon:
        idxs_choice.update(np.random.choice(np.array(list(final_fac_idxs.difference(idxs_choice))),
                                            budgets[t], replace=False))
        y_ws[t] = [final_fac[i] for i in idxs_choice]

    y_val = {t: [1 if j in y_ws[t] else 0 for j in node_list] for t in time_horizon}

    print('\t RANDOM WARM START:: {v1}'.format(v1=(time.time() - t0)))

    return y_val, seed


def y_warm_start_greedy(A_out, H_out, pod_comb, A_rev, H_rev, pdo_comb, E_out, E_rev,
                        node_list, facility_costs, budgets, time_horizon, pod_flows, suppress_output=True):
    """
    Solve and plot solution for path constrained facility location problem

    For infeasible stretches
    -Can facility set to be priority set nodes and include all 'Other' type nodes as clients in the future
    -Find shortest path from all priority set nodes to ALL nodes (priority set + 'Other') using nx.multi_source_dijkstra
    -Only allow facilities to be placed at priority set nodes and allow for additional variables z_j to mark infeasible
        locations for each of the 'Other' nodes not satisfied.
    :param tolerance:
    :param rr:
    :param all_pairs:
    :param intertypes:
    :param D: [float] range in km
    :param binary_prog:
    :param path_cnstrs_only:
    :param plot:
    :param plot_paths: for plotting only paths
    :param origin:
    :return:
    """

    n = len(node_list)
    # set up and run backwards myopic model (based on Chung and Kwon, 2015)
    cum_budgets = {t_step: sum(budgets[t] for t in time_horizon[:t_idx + 1])
                   for t_idx, t_step in enumerate(time_horizon)}
    y_val = {t: [] for t in time_horizon}
    # z_val = {t: [] for t in time_horizon}
    # w_val = {t: [] for t in time_horizon}
    # obj_val = {t: [] for t in time_horizon}
    t_future = -1
    iter_count = 1
    for t in list(reversed(time_horizon)):
        # set up model
        m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(suppress_output))
        # facility upper bounds (based on fixed facilities)
        if t_future == -1:
            ub = np.ones((n,))
        else:
            ub = y_val[t_future]
        # facility location and flow selection variables
        y = m.addMVar((n,), vtype=GRB.BINARY, ub=ub, name='y_{v1}'.format(v1=t))
        z = m.addMVar((len(pod_comb),), lb=0, ub=1, name='z_{v1}'.format(v1=t))
        w = m.addMVar((len(pdo_comb),), lb=0, ub=1, name='w_{v1}'.format(v1=t))

        # objective fxn
        m.setObjective(pod_flows[t].transpose() @ z, GRB.MAXIMIZE)

        # constraints
        # outbound path facility-flow coverage
        m.addConstr(A_out @ y >= H_out @ z, name='C2_{v1}'.format(v1=t))
        # return path facility-flow coverage
        m.addConstr(A_rev @ y >= H_rev @ w, name='C3_{v1}'.format(v1=t))
        # outbound-return path-od relation
        m.addConstr(E_out @ z <= E_rev @ w, name='C4_{v1}'.format(v1=t))
        # outbound path-od relation (redundant - can remove)
        # m.addConstr(E_out @ z[t] <= np.ones((E_out.shape[0],)), name='C5_{v1}'.format(v1=t))
        # return path-od relation
        m.addConstr(E_rev @ w <= np.ones((E_rev.shape[0],)), name='C6_{v1}'.format(v1=t))

        # budget constraints (cumulative here)
        m.addConstr(facility_costs[t] @ y <= cum_budgets[t], name='C7_{v1}'.format(v1=t))

        # optimize
        m.update()
        m.optimize()

        # extract solution values
        y_val[t] = y.X
        # z_val[t] = z.X
        # w_val[t] = w.X
        # obj_val[t] = m.ObjVal  # get objective fxn value

        # update for next step
        t_future = t
        iter_count += 1

    return y_val


def mp_dc_frlm_max_greedy_initialization(A_out, H_out, pod_comb, A_rev, H_rev, pdo_comb, E_out, E_rev, node_list,
                                         fac_costs, time_horizon, pod_flows, budgets,
                                         fixed_facilities: dict = None, barred_facilities: dict = None,
                                         warm_start_strategy: str = None, suppress_output=True):

    n = len(node_list)

    y_ws = None
    y_lb = {t: np.zeros((n,)) for t in time_horizon}
    y_ub = {t: np.ones((n,)) for t in time_horizon}

    if warm_start_strategy == 'greedy' or fixed_facilities == 'static':
        y_greedy = y_warm_start_greedy(A_out, H_out, pod_comb, A_rev, H_rev, pdo_comb, E_out, E_rev, node_list,
                                       fac_costs, budgets, time_horizon, pod_flows, suppress_output=suppress_output)
        if warm_start_strategy == 'greedy':
            y_ws = y_greedy

        if fixed_facilities == 'static':
            y_lb[time_horizon[-1]] = np.array([1 if y_greedy[time_horizon[-1]][j] == 1 else 0 for j in range(n)])

    for t in time_horizon:
        if isinstance(fixed_facilities, dict):
            if t in fixed_facilities.keys():
                y_lb[t] = np.array([1 if j in fixed_facilities[t] else 0 for j in node_list])
        if isinstance(barred_facilities, dict):
            if t in barred_facilities.keys():
                y_ub[t] = np.array([0 if j in barred_facilities[t] else 1 for j in node_list])

    return y_ws, y_lb, y_ub


def convert_np_arrays_dc_mp(n_arrays):
    '''
    Arrays in n_arrays are indexed by:
    n_arrays['outbound_A'] -> A,
            A has rows corresponding to the values in row_idxs_A, cols corresponding to the values in col_idxs_A
        'return_B' -> B,
            B has rows corresponding to the values in row_idxs_B, cols corresponding to the values in col_idxs_B
        'row_idxs_A' -> row_idxs_A,
            Contains arrays of entries corresponding to rows of A:
            i.e., row_idxs_A[0] = [i, phi_idx, o, d] which corresponds to the combination (i, phi_idx, (o, d))
        'row_idxs_B' -> row_idxs_B,
            See above
        'row_idxs_name_type_AB' -> row_idxs_name_type_AB,
            First row contains names of entries: ['i_nodeid', 'phi_idx', 'orig', 'dest']
            Second row contains types of entries: ['str', 'int', 'str', 'str']
        'col_idxs_AB1' -> col_idxs_A,
            Contains arrays of entries corresponding to cols of A for j_idx < n = len(node_list):
            i.e., col_idxs_A[j_idx] = j, which corresponds to node j in node_list
        'col_idxs_A2' -> col_idxs_A2,
            Contains arrays of entries corresponding to cols of A for n <= j_idx < n + p:
            i.e., col_idxs_A[j_idx] = [phi_idx, o, d], which correspond to combination (phi_idx, (o, d))
        'col_idxs_B2' -> col_idxs_B2,
            See above
        'col_idxs_name_type_AB' -> col_idxs_name_type_AB
            First row contains names of entries: ['phi_idx', 'orig', 'dest']
            Second row contains types of entries: ['int', 'str', 'str']

    Parameters
    ----------
    n_arrays

    Returns
    -------

    '''

    A_out = n_arrays['A_out'].tolist()
    H_out = n_arrays['H_out'].tolist()
    A_rev = n_arrays['A_rev'].tolist()
    H_rev = n_arrays['H_rev'].tolist()
    npod_comb = [(i, int(phi_idx), int(rev_idx), (o, d)) for i, phi_idx, rev_idx, o, d in n_arrays['row_idxs_A_out']]
    npdo_comb = [(i, int(phi_idx), int(rev_idx), (d, o)) for i, phi_idx, rev_idx, d, o in n_arrays['row_idxs_A_rev']]
    node_list = n_arrays['col_idxs_A']
    pod_comb = [(int(phi_idx), (o, d)) for phi_idx, o, d in n_arrays['col_idxs_H_out']]
    pdo_comb = [(int(phi_idx), (d, o)) for phi_idx, d, o in n_arrays['col_idxs_H_rev']]

    return A_out, H_out, npod_comb, pod_comb, A_rev, H_rev, npdo_comb, pdo_comb, node_list


def convert_np_arrays_time_dc_mp(n_arrays, range_km: list):
    '''
    Arrays in n_arrays are indexed by:
    n_arrays['outbound_A'] -> A,
            A has rows corresponding to the values in row_idxs_A, cols corresponding to the values in col_idxs_A
        'return_B' -> B,
            B has rows corresponding to the values in row_idxs_B, cols corresponding to the values in col_idxs_B
        'row_idxs_A' -> row_idxs_A,
            Contains arrays of entries corresponding to rows of A:
            i.e., row_idxs_A[0] = [i, phi_idx, o, d] which corresponds to the combination (i, phi_idx, (o, d))
        'row_idxs_B' -> row_idxs_B,
            See above
        'row_idxs_name_type_AB' -> row_idxs_name_type_AB,
            First row contains names of entries: ['i_nodeid', 'phi_idx', 'orig', 'dest']
            Second row contains types of entries: ['str', 'int', 'str', 'str']
        'col_idxs_AB1' -> col_idxs_A,
            Contains arrays of entries corresponding to cols of A for j_idx < n = len(node_list):
            i.e., col_idxs_A[j_idx] = j, which corresponds to node j in node_list
        'col_idxs_A2' -> col_idxs_A2,
            Contains arrays of entries corresponding to cols of A for n <= j_idx < n + p:
            i.e., col_idxs_A[j_idx] = [phi_idx, o, d], which correspond to combination (phi_idx, (o, d))
        'col_idxs_B2' -> col_idxs_B2,
            See above
        'col_idxs_name_type_AB' -> col_idxs_name_type_AB
            First row contains names of entries: ['phi_idx', 'orig', 'dest']
            Second row contains types of entries: ['int', 'str', 'str']

    Parameters
    ----------
    n_arrays

    Returns
    -------

    '''

    A_out = {r_t: n_arrays[f'A_out_{r_t}'].tolist() for r_t in range_km}
    H_out = n_arrays['H_out'].tolist()
    A_rev = {r_t: n_arrays[f'A_rev_{r_t}'].tolist() for r_t in range_km}
    H_rev = n_arrays['H_rev'].tolist()
    npod_comb = [(i, int(phi_idx), int(rev_idx), (o, d)) for i, phi_idx, rev_idx, o, d in n_arrays['row_idxs_A_out']]
    npdo_comb = [(i, int(phi_idx), int(rev_idx), (d, o)) for i, phi_idx, rev_idx, d, o in n_arrays['row_idxs_A_rev']]
    node_list = n_arrays['col_idxs_A']
    pod_comb = [(int(phi_idx), (o, d)) for phi_idx, o, d in n_arrays['col_idxs_H_out']]
    pdo_comb = [(int(phi_idx), (d, o)) for phi_idx, d, o in n_arrays['col_idxs_H_rev']]

    return A_out, H_out, npod_comb, pod_comb, A_rev, H_rev, npdo_comb, pdo_comb, node_list

