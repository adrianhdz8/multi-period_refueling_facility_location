from util import *
# MODULES
from helper import splc_to_node, node_to_edge_path, extract_rr, load_dict_from_json, write_dict_to_json

'''
ROUTING METHODS
'''


def route_flows_mp(G: nx.DiGraph, time_horizon: list, od_flows_tons: dict):

    edge_list = list(G.edges)
    edge_list.extend([(v, u) for u, v in edge_list])  # both directions00
    od_list = list(od_flows_tons.keys())
    od_flow_values = {t: np.array([od_flows_tons[od][t] for od in od_list]) for t in time_horizon}
    t0 = time.time()
    pli_mat = path_link_incidence_mat_mp(G=G, od_list=od_list, edge_list=edge_list)
    # print('PLI MATRIX:: {v0} seconds'.format(v0=time.time() - t0))

    for e in edge_list:
        # instantiate dict storage objects
        G.edges[e]['baseline_avg_ton'] = {t: 0 for t in time_horizon}
        G.edges[e]['alt_tech_avg_ton'] = {t: 0 for t in time_horizon}
        G.edges[e]['support_diesel_avg_ton'] = {t: 0 for t in time_horizon}

    # flow assignment by <time_period>
    for t in time_horizon:
        t1 = time.time()
        selected_ods = set(G.graph['framework']['selected_ods'][t])
        selected_ods = selected_ods.union({(d, o) for o, d in selected_ods})
        # print(len(od_flow_values[t]))
        # print(od_flow_values[t])
        # print(pli_mat.shape)

        edge_tons = pli_mat @ od_flow_values[t]
        # Alt. Tech.
        # determine which OD pairs are captured - set flows of all those not captured to 0
        od_flows_capt = np.array([f if od_list[od_idx] in selected_ods else 0
                                  for od_idx, f in enumerate(od_flow_values[t])])
        edge_tons_alt_tech = pli_mat @ od_flows_capt
        # Support Diesel
        # complement of captured flows, i.e., baseline - alt. tech.
        od_flows_not_capt = np.array([0 if od_list[od_idx] in selected_ods else f
                                      for od_idx, f in enumerate(od_flow_values[t])])
        edge_tons_sd = pli_mat @ od_flows_not_capt
        for e_idx, e_ton in enumerate(edge_tons):
            if e_ton == 0:
                continue
            e = edge_list[e_idx]
            # Baseline:
            G.edges[e]['baseline_avg_ton'][t] = e_ton
            # ----------
            # Alt. Tech.
            # determine which OD pairs are captured - set flows of all those not captured to 0
            e_ton_alt_tech = edge_tons_alt_tech[e_idx]  # edit
            G.edges[e]['alt_tech_avg_ton'][t] = e_ton_alt_tech
            # --------------
            # Support Diesel
            # complement of captured flows, i.e., baseline - alt. tech.
            e_ton_sd = edge_tons_sd[e_idx]
            G.edges[e]['support_diesel_avg_ton'][t] = e_ton_sd
        # print('\t EDGE ASSIGNMENT {v0}:: {v1} seconds'.format(v0=t, v1=time.time() - t1))

    # all values here are annual
    G.graph['operations'] = dict(
        baseline_total_annual_tonmi={t: sum(G.edges[e]['baseline_avg_ton'][t] * G.edges[e]['miles'] for e in edge_list)
                                     for t in time_horizon},
        alt_tech_total_annual_tonmi={t: sum(G.edges[e]['alt_tech_avg_ton'][t] * G.edges[e]['miles']
                                            for e in edge_list) for t in time_horizon},
        support_diesel_total_annual_tonmi={t: sum(G.edges[e]['support_diesel_avg_ton'][t] * G.edges[e]['miles']
                                                  for e in edge_list) for t in time_horizon},
    )

    G.graph['operations'].update(dict(
        deployment_perc={t: (G.graph['operations']['alt_tech_total_annual_tonmi'][t] /
                             G.graph['operations']['baseline_total_annual_tonmi'][t]) for t in time_horizon},
    ))

    return G


def path_link_incidence_mat_mp(G: nx.DiGraph, od_list: list, edge_list: list):
    # TODO: see notes on generating
    # od_flows is time-indexed as well now; want union of all ods with >0 flow
    # od_list = list(od_flows.keys())

    edge_idx_dict = {v: i for i, v in enumerate(edge_list)}

    # precompute shortest path for all OD pairs (since Dijkstra's finds one-to-all shortest paths)
    sp_dict = dict(nx.all_pairs_dijkstra_path(G, weight='km'))
    pli_data = []
    pli_rows = []
    pli_cols = []

    for od_idx, (o, d) in enumerate(od_list):
        path_edges = node_to_edge_path(sp_dict[o][d])
        pli_data.extend([1 for _ in range(len(path_edges))])
        pli_rows.extend([edge_idx_dict[e] for e in path_edges])
        pli_cols.extend([od_idx for _ in range(len(path_edges))])

    pli_mat = csr_matrix((pli_data, (pli_rows, pli_cols)), shape=(len(edge_list), len(od_list)))

    return pli_mat


'''
OD FLOWS
'''


def ods_by_perc_ton_mi_mp(G: nx.DiGraph, perc_ods: float, constant_flows=False, od_flows_truncate=False):
    # return O-D pairs in CCWS tha provide ton flows >= <perc_ods> * total CCWS ton flows
    # od_flows is average daily ton-miles

    # load dict that maps SPLC codes to node_ids in G
    splc_node_dict = splc_to_node(G)
    # load grouped OD flow data
    # flow_df = RR_SPLC_comm_grouping(filename=CCWS_filename, time_window=time_window)
    flow_df = pd.read_csv(os.path.join(FAF5_DIR, 'WB2019_summary_class1_SPLC_forecast_2050.csv'),
                          header=0, index_col=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name'])

    # filter out specific railroad
    rr = G.graph['railroad']
    if 'KCS_ex' in rr:
        # for KCS example network
        flow_df = extract_rr(flow_df, 'KCS', forecast=True)
    else:
        flow_df = extract_rr(flow_df, rr, forecast=True)
    # only index needed is the OD pair
    flow_df.reset_index(level='Commodity Group Name', inplace=True)
    # filter out OD pairs that are not in the splc_node_dict keys
    splc_set = set(splc_node_dict.keys())
    remove_idxs = list({i for i in flow_df.index.unique() if i[1:7] not in splc_set or i[7:] not in splc_set})
    flow_df.drop(index=remove_idxs, inplace=True)

    # assign each SPLC OD to its respective nodeid in G
    flow_df.reset_index(level='Origin-Destination SPLC', inplace=True)
    flow_df['Origin-Destination nodeid'] = flow_df['Origin-Destination SPLC'].apply(lambda x:
                                                                                    (splc_node_dict[x[1:7]],
                                                                                     splc_node_dict[x[7:]]))

    years = ['', '2020 ', '2023 ', '2025 ', '2030 ', '2035 ', '2040 ', '2045 ', '2050 ']
    cols_to_keep = [y + 'Expanded Tons' for y in years]
    flow_df['Origin-Destination nodeid comb'] = flow_df['Origin-Destination nodeid'].apply(lambda x: x[0] + x[1])
    comb_od_nodeid_dict = {flow_df.loc[i, 'Origin-Destination nodeid comb']:
                               flow_df.loc[i, 'Origin-Destination nodeid'] for i in flow_df.index}
    # od_nodeid_comb_dict = {(o, d): o + d for o, d in flow_df.index}
    flow_df = flow_df.groupby(by=['Origin-Destination nodeid comb']).sum(numeric_only=True)[cols_to_keep]
    flow_df['Origin-Destination nodeid comb'] = flow_df.index

    key = 'Expanded Ton-Miles Routed'
    # tons = flow_df['Expanded Tons'].to_dict()
    # load from json or compute if does not exist
    filepath_sp_dict = os.path.join(SHORTEST_PATH_DIR, rr + '_SP_dict_miles.json')
    if os.path.exists(filepath_sp_dict):
        miles = load_dict_from_json(filepath_sp_dict)
    else:
        miles = dict(nx.all_pairs_bellman_ford_path_length(G=G, weight='miles'))
        write_dict_to_json(miles, filepath_sp_dict)
    for y in years:
        flow_df[y + 'Expanded Ton-Miles Routed'] = flow_df['Origin-Destination nodeid comb'].apply(
            lambda x: (flow_df.loc[x, y + 'Expanded Tons'] *
                       miles[comb_od_nodeid_dict[x][0]][comb_od_nodeid_dict[x][1]]))
    # flow_df.drop(columns=['Origin-Destination nodeid comb'] + cols_to_keep, inplace=True)
    flow_df.drop(columns=['Origin-Destination nodeid comb'], inplace=True)

    # group by OD pair nodeid, summing all commodity groupings for the total ton-mile values (over all commodities)
    # keep only dataframe with ton-miles sum
    # flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Expanded Ton-Miles Routed']]
    # sort OD pairs by ton-miles or tons in descending order
    flow_df.sort_values(by=key, ascending=False, inplace=True)

    # compute cumulative percentage of the ton-miles or tons
    flow_df['Cumulative Percent'] = (flow_df[key].cumsum() / flow_df[key].sum())
    # select the subset of OD pairs that provides a cumulative percentage of ton-miles or tons >= <perc_ods>
    m = flow_df[flow_df['Cumulative Percent'] >= perc_ods]['Cumulative Percent'].min()
    if m is np.NAN:
        m = 1
    ods = flow_df[flow_df['Cumulative Percent'] <= m].index
    # convert OD pair strings into node_id pair tuples
    # get O-D flows for all O-D pairs as a dict
    if od_flows_truncate:
        flow_df = flow_df.loc[ods]

    ods = [comb_od_nodeid_dict[od] for od in ods]
    ods = [(o, d) for o, d in ods if o != d]

    flow_df.rename(index=comb_od_nodeid_dict, inplace=True)
    flow_df.fillna(0, inplace=True)
    if constant_flows:
        od_flows = flow_df[key].to_dict()
        od_flows_tons = flow_df['Expanded Tons'].to_dict()
        if od_flows_truncate:
            od_flows = {od: od_flows[od] for od in ods}
            od_flows_tons = {od: od_flows_tons[od] for od in ods}
    else:
        year_mapper = {'2019': '', '2020': '2020 ', '2023': '2023 ', '2025': '2025 ', '2030': '2030 ', '2035': '2035 ',
                       '2040': '2040 ', '2045': '2045 ', '2050': '2050 '}
        od_flows = {y: [] for y in year_mapper.keys()}
        od_flows_tons = {y: [] for y in year_mapper.keys()}
        for y, y_name in year_mapper.items():
            od_flows[y] = flow_df[y_name + key].to_dict()
            od_flows_tons[y] = flow_df[y_name + 'Expanded Tons'].to_dict()

        od_flows = {od: {y: od_flows[y][od] if y in od_flows.keys() and od in od_flows[y].keys() else 0
                         for y in year_mapper.keys()} for od in ods}
        od_flows_tons = {od: {y: od_flows_tons[y][od] if (y in od_flows_tons.keys()
                                                          and od in od_flows_tons[y].keys()) else 0
                              for y in year_mapper.keys()} for od in ods}

    return ods, od_flows, od_flows_tons
