from util import *
# MODULES
from network_processing import load_simplified_consolidated_graph
# from facility_location import facility_location_mp
from facility_location_oop import facility_location_mp
from routing import ods_by_perc_ton_mi_mp, route_flows_mp
from plotting import plot_battery_facility_location
from input_output import extract_assert_scenario_mp_inputs, extract_assert_param_inputs


# TODO: simplify to only focus on necessary facility location problems and any required post-processing
#  (i.e., routing, emissions/cost estimation, solution statistics, and plotting)


# TODO: make so that it can handle static and dynamic cases
def run_rollout(rr: str, range_km: dict, time_horizon: list, facility_costs: dict = None,
                max_flow=False, greedy=False, flow_mins: float = None, budgets: dict = None,
                candidate_facilities: list = None, discount_rates: any = None, num_shortest_paths: int = 1,
                od_flow_perc: float = 1, deviation_paths=True,
                fixed_facilities: dict = None, barred_facilities: dict = None,
                y_warm_start: dict = None, warm_start_strategy: str = None,
                solution_tol: float = None, strong_cuts=True, nested=True, od_flows_provided=False,
                benders=False, multi_cut_time=False, multi_cut_ods=False, analytic_sp=True,
                solution_aids=False, solution_time_limit=None,
                max_iter=None,
                binary_prog=True, suppress_output=False,
                constant_flows=False, title: str = None, G: nx.DiGraph = None,
                plot=True, colors=False, time_step_label=True
                ):
    if not G:
        G = load_simplified_consolidated_graph(rr)

    G.graph['scenario'] = dict(railroad=rr, range_mi={t: range_km[t] * KM2MI for t in time_horizon}, range_km=range_km)
    t0 = time.time()
    if od_flows_provided:
        od_flows = G.graph['od_flows']
        if 'od_ton_flows' in G.graph.keys():
            od_flows_tons = G.graph['od_ton_flows']
        else:
            od_flows_tons = od_flows
    else:
        # select almost all O-D pairs with non-zero flow (leave out the X% with the lowest flow values; too many)
        ods, od_flows, od_flows_tons = ods_by_perc_ton_mi_mp(G=G, perc_ods=od_flow_perc,
                                                             constant_flows=constant_flows,
                                                             od_flows_truncate=True)

    print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    # 2. locate facilities and extract graph form of this, G
    G, time_limit_reached = facility_location_mp(G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                                                 facility_costs=facility_costs,
                                                 max_flow=max_flow, greedy=greedy, flow_mins=flow_mins, budgets=budgets,
                                                 discount_rates=discount_rates,
                                                 fixed_facilities=fixed_facilities, barred_facilities=barred_facilities,
                                                 y_warm_start=y_warm_start, warm_start_strategy=warm_start_strategy,
                                                 solution_tol=solution_tol, strong_cuts=strong_cuts, nested=nested,
                                                 deviation_paths=deviation_paths,
                                                 num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                                                 benders=benders, multi_cut_time=multi_cut_time,
                                                 multi_cut_ods=multi_cut_ods,
                                                 analytic_sp=analytic_sp,
                                                 solution_aids=solution_aids, solution_time_limit=solution_time_limit,
                                                 max_iter=max_iter,
                                                 binary_prog=binary_prog, suppress_output=suppress_output)
    print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    # 3. assign flows to links and get average ton and locomotive flows for each edge
    G = route_flows_mp(G=G, time_horizon=time_horizon, od_flows_tons=od_flows_tons)
    print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

    # update stats
    # G = operations_stats_mp(G=G, time_horizon=time_horizon)

    t0 = time.time()
    # TODO: make so that it can handle static and dynamic cases
    if plot:
        fig = plot_battery_facility_location(G, time_horizon=time_horizon, additional_plots=True, nested=nested,
                                             max_flow=max_flow, colors=colors, time_step_label=time_step_label,
                                             title=title)
    else:
        fig = None
    print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))

    return G, fig


def run_rollout_file(scenario_code: str, solver_param_code: str):
    # solver params: greedy, solution_aids, benders, strong_cuts, multi_cut, analytic_sp,
    #  warm_start_strategy, max_iter, solution_tol, solution_time_limit, binary_prog, suppress_output,

    [rr, time_horizon, range_km, budgets, max_flow, flow_mins, discount_rates, num_shortest_paths,
     facility_costs, fixed_facilities, barred_facilities,
     od_flow_perc, deviation_paths, nested, constant_flows, od_flows_provided,
     title, plot, colors, time_step_label] = extract_assert_scenario_mp_inputs(scenario_code=scenario_code)

    [greedy, solution_aids, benders, strong_cuts, multi_cut_time, multi_cut_ods, analytic_sp,
     warm_start_strategy, max_iter, solution_tol, solution_time_limit,
     binary_prog, suppress_output, print_status] = extract_assert_param_inputs(solver_param_code=solver_param_code)

    G = load_simplified_consolidated_graph(rr)

    G.graph['scenario'] = dict(railroad=rr, range_mi={t: range_km[t] * KM2MI for t in time_horizon}, range_km=range_km)
    t0 = time.time()
    if od_flows_provided:
        od_flows = G.graph['od_flows']
        if 'od_ton_flows' in G.graph.keys():
            od_flows_tons = G.graph['od_ton_flows']
        else:
            od_flows_tons = od_flows
    else:
        # select almost all O-D pairs with non-zero flow (leave out the X% with the lowest flow values; too many)
        ods, od_flows, od_flows_tons = ods_by_perc_ton_mi_mp(G=G, perc_ods=od_flow_perc,
                                                             constant_flows=constant_flows,
                                                             od_flows_truncate=True)

    if print_status:
        print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    # 2. locate facilities and extract graph form of this, G
    # print(f'|E|={G.number_of_edges()}, |V|={G.number_of_nodes()}, |Q|={len(od_flows)}')
    G, time_limit_reached = facility_location_mp(G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                                                 facility_costs=facility_costs,
                                                 max_flow=max_flow, greedy=greedy, flow_mins=flow_mins, budgets=budgets,
                                                 discount_rates=discount_rates,
                                                 fixed_facilities=fixed_facilities, barred_facilities=barred_facilities,
                                                 warm_start_strategy=warm_start_strategy,
                                                 solution_tol=solution_tol, strong_cuts=strong_cuts, nested=nested,
                                                 deviation_paths=deviation_paths,
                                                 num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                                                 benders=benders, multi_cut_time=multi_cut_time,
                                                 multi_cut_ods=multi_cut_ods, analytic_sp=analytic_sp,
                                                 solution_aids=solution_aids, solution_time_limit=solution_time_limit,
                                                 max_iter=max_iter,
                                                 binary_prog=binary_prog, suppress_output=suppress_output,
                                                 print_status=print_status,
                                                 scenario_code=scenario_code, solver_param_code=solver_param_code)

    if time_limit_reached:
        return G, None

    if print_status:
        print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    # 3. assign flows to links and get average ton and locomotive flows for each edge
    G = route_flows_mp(G=G, time_horizon=time_horizon, od_flows_tons=od_flows_tons)
    if print_status:
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

    # update stats
    # G = operations_stats_mp(G=G, time_horizon=time_horizon)

    t0 = time.time()
    if plot:
        fig = plot_battery_facility_location(G, time_horizon=time_horizon, additional_plots=True, nested=nested,
                                             max_flow=max_flow, colors=colors, time_step_label=time_step_label,
                                             title=title)
    else:
        fig = None

    if print_status:
        print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))

    return G, fig


def run_multiple_scenarios(scenario_codes: list, solver_param_files: list):
    # run multiple scenarios with different solution methods,
    # <solver_param_files> is a list of lists corresponding to the list of solution methods to use for that scenario

    assert len(scenario_codes) == len(solver_param_files), \
        '<scenario_codes> and <solver_param_files> must be of the same length.'

    total_num = sum(len(s) for s in solver_param_files)
    count = 0
    for i, scenario_code in enumerate(scenario_codes):
        for solver_param_file in solver_param_files[i]:
            print(f'RUNNING:: <{scenario_code}> with <{solver_param_file}>', end=' ')
            # # getting a SIGKILL error here
            # if (scenario_code == 'sc_usa1_rt1600_b5_od0.9_r0.1_lnghrz_k1' and
            #         solver_param_file == 'sp_bd_mcon_scon_0.001_3600_large'):
            #     count += 1
            #     print(f'=> {count}/{total_num} COMPLETE')
            #     continue

            if not os.path.exists(os.path.join(EXP_DIR, scenario_code + '_' + solver_param_file + '.csv')):
                _, _ = run_rollout_file(scenario_code, solver_param_file)

            # _, _ = run_rollout_file(scenario_code, solver_param_file)
            count += 1
            print(f'=> {count}/{total_num} COMPLETE')
    print('ALL COMPLETE')
    print('============')


def operations_stats_mp(G: nx.DiGraph, time_horizon: list) -> nx.DiGraph:
    # compute the operational stats of solution in G (many relative to diesel baseline)

    comm_list = list(G.graph['operations']['baseline_total_annual_tonmi'][time_horizon[0]].keys())

    # G.graph['operations'].update(
    #     dict(
    #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] -
    #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
    #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group]),
    #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'] -
    #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
    #                                         (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'] -
    #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
    #     )
    # )
    G.graph['operations'].update(dict(
        emissions_change={t: {c: 100 * (G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c] -
                                        G.graph['energy_source_LCA'][t]['annual_total_emissions_tonco2'][c]) /
                                 G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c]
                              for c in comm_list} for t in time_horizon},
        cost_avoided_emissions={t: {c: 1e-3 * (G.graph['energy_source_TEA'][t]['total_scenario_LCO_tonmi'][c] -
                                               G.graph['diesel_TEA'][t]['total_LCO_tonmi'][c]) /
                                       (G.graph['energy_source_LCA'][t]['avg_emissions_tonco2_tonmi'][c] -
                                        G.graph['diesel_LCA'][t]['total_emissions_tonco2_tonmi'][c])
                                    for c in comm_list} for t in time_horizon},
        # dict(zip(
        #     comm_list,
        #         [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
        #                   G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
        #          (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
        #           G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list])),
        # cost_avoided_emissions_no_delay=dict(zip(
        #     comm_list,
        #     [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][c] -
        #               G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
        #      (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
        #       G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
    ))
    # elif G.graph['scenario']['fuel_type'] == 'hydrogen':
    #     # G.graph['operations'].update(
    #     #     dict(
    #     #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'] -
    #     #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
    #     #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2']),
    #     #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'] -
    #     #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
    #     #                                         (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'] -
    #     #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
    #     #     )
    #     # )
    #     G.graph['operations'].update(
    #         dict(
    #             emissions_change=dict(zip(
    #                 comm_list,
    #                 [100 * (G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] -
    #                         G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][c]) /
    #                  G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] for c in comm_list])),
    #             cost_avoided_emissions=dict(zip(
    #                 comm_list,
    #                 [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
    #                           G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
    #                  (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
    #                   G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list])),
    #             cost_avoided_emissions_no_delay=dict(zip(
    #                 comm_list,
    #                 [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
    #                           G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
    #                  (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
    #                   G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
    #         ))
    # elif G.graph['scenario']['fuel_type'] == 'e-fuel' or G.graph['scenario']['fuel_type'] == 'biodiesel':
    #     # G.graph['operations'].update(
    #     #     dict(
    #     #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'] -
    #     #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
    #     #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2']),
    #     #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_LCO_tonmi'] -
    #     #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
    #     #                                         (G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'] -
    #     #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
    #     #     )
    #     # )
    #     G.graph['operations'].update(
    #         dict(
    #             emissions_change=dict(zip(
    #                 comm_list,
    #                 [100 * (G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] -
    #                         G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][c]) /
    #                  G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] for c in comm_list])),
    #             cost_avoided_emissions=dict(zip(
    #                 comm_list,
    #                 [-1e-3 * (G.graph['energy_source_TEA']['total_LCO_tonmi'][c] -
    #                           G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
    #                  (G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'][c] -
    #                   G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
    #         ))

    return G

# G, f, l = run_scenario('BNSF', 'battery', perc_ods=0.4, D=600*1.6, plot=True,
#                        load_scenario=False, deployment_table=True)


# def run_scenario(rr: str = None, fuel_type: str = None, deployment_perc: float = None,
#                  D: float = None, reroute: bool = None, switch_tech: bool = None, max_reroute_inc: float = None,
#                  max_util: float = None, station_type: str = None, h2_fuel_type: str = None,
#                  clean_energy=False, clean_energy_cost: float = None, emissions_obj=False,
#                  CCWS_filename: str = None, perc_ods: float = None, comm_group: str = 'TOTAL',
#                  time_window: tuple = None, freq: str = 'M',
#                  eff_energy_p_tender: float = None, tender_cost_p_tonmi: float = None, diesel_cost_p_gal: float = None,
#                  suppress_output=True, binary_prog=True, select_cycles=False, max_flow=False, budget: int = None,
#                  deviation_paths=True, extend_graph=True, od_flow_perc: float = 1,
#                  G: nx.DiGraph = None, radius: float = None, intertypes: set = None,
#                  scenario_code: str = None, deployment_table=False,
#                  plot=True, load_scenario=True, cache_scenario=False, legend_show=True):
#     # deployment_table = True means we are creating the deployment table and bypass the deployment_table lookup
#
#     if not scenario_code:
#         df_scenario = write_scenario_df(rr, fuel_type, deployment_perc,
#                                         D, reroute, switch_tech, max_reroute_inc,
#                                         max_util, station_type,
#                                         clean_energy, clean_energy_cost, emissions_obj,
#                                         CCWS_filename, perc_ods, comm_group,
#                                         time_window, freq,
#                                         eff_energy_p_tender,
#                                         suppress_output, binary_prog,
#                                         radius, intertypes, deployment_table)
#         scenario_code = df_scenario['scenario_code']
#     else:
#         df_scenario = load_scenario_df(scenario_code=scenario_code)
#
#     if not deployment_table and load_scenario:
#         t0 = time.time()
#         if cache_exists(scenario_code):
#             G = load_cached_graph(scenario_code)
#             print('LOADED GRAPH FROM CACHE:: %s seconds ---' % round(time.time() - t0, 3))
#         else:
#             G = run_scenario_df(df_scenario=df_scenario, G=G,
#                                 select_cycles=select_cycles, deviation_paths=deviation_paths,
#                                 max_flow=max_flow, budget=budget,
#                                 extend_graph=extend_graph, od_flow_perc=od_flow_perc)
#             if cache_scenario:
#                 t0 = time.time()
#                 cache_graph(G=G, scenario_code=scenario_code)
#                 print('CACHE GRAPH:: %s seconds ---' % round(time.time() - t0, 3))
#     else:
#         if deployment_table:
#             df_scenario['perc_ods'] = perc_ods
#         G = run_scenario_df(df_scenario=df_scenario, G=G,
#                             select_cycles=select_cycles, deviation_paths=deviation_paths,
#                             max_flow=max_flow, budget=budget, extend_graph=extend_graph,
#                             od_flow_perc=od_flow_perc)
#
#         if cache_scenario:
#         # if False:
#             t0 = time.time()
#             cache_graph(G=G, scenario_code=scenario_code)
#             print('CACHE GRAPH:: %s seconds ---' % round(time.time() - t0, 3))
#
#     G = update_graph_values(G=G, fuel_type=df_scenario['fuel_type'], max_util=df_scenario['max_util'],
#                             station_type=df_scenario['station_type'],
#                             clean_energy=clean_energy, clean_energy_cost=clean_energy_cost,
#                             h2_fuel_type=h2_fuel_type, tender_cost_p_tonmi=tender_cost_p_tonmi,
#                             diesel_cost_p_gal=diesel_cost_p_gal)
#
#     if plot:
#         t0 = time.time()
#         fig, label = plot_scenario(G, fuel_type=df_scenario['fuel_type'],
#                                    deployment_perc=df_scenario['deployment_perc'],
#                                    comm_group=df_scenario['comm_group'], legend_show=legend_show)
#         print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))
#     else:
#         fig = None
#         label = None
#
#     return G, fig, label


# def run_scenario_df(df_scenario: pd.DataFrame, G: nx.DiGraph = None,
#                     select_cycles=False, deviation_paths=True,
#                     max_flow=False, budget: int = None, extend_graph=True,
#                     od_flow_perc: float = 1):
#     t0_total = time.time()
#
#     if isinstance(df_scenario, pd.DataFrame):
#         df_scenario = df_scenario['Value']
#
#     idxs = ['rr', 'fuel_type', 'deployment_perc',
#             'D', 'reroute', 'switch_tech', 'max_reroute_inc',
#             'max_util', 'station_type',
#             'clean_energy', 'clean_energy_cost', 'emissions_obj',
#             'CCWS_filename', 'perc_ods', 'comm_group',
#             'time_window_start', 'time_window_end', 'freq',
#             'eff_energy_p_tender',
#             'suppress_output', 'binary_prog',
#             'radius', 'intertypes',
#             'scenario_code']
#
#     [rr, fuel_type, deployment_perc,
#      D, reroute, switch_tech, max_reroute_inc,
#      max_util, station_type,
#      clean_energy, _, emissions_obj,
#      CCWS_filename, perc_ods, comm_group,
#      time_window_start, time_window_end, freq,
#      eff_energy_p_tender,
#      suppress_output, binary_prog,
#      radius, intertypes,
#      scenario_code] = df_scenario.reindex(idxs)
#
#     # deployment_perc = float(deployment_perc)
#     # D = float(D)
#     # reroute = int(reroute)
#     # switch_tech = int(switch_tech)
#     # max_reroute_inc = float(max_reroute_inc)
#     # max_util = float(max_util)
#     # clean_energy = int(clean_energy)
#     # clean_energy_cost = float(clean_energy_cost)
#     # emissions_obj = int(emissions_obj)
#     # eff_energy_p_tender = float(eff_energy_p_tender)
#     # suppress_output = int(suppress_output)
#     # binary_prog = int(binary_prog)
#     # radius = float(radius)
#
#     time_window = (time_window_start, time_window_end)
#
#     # 0. load railroad network representation as a nx.Graph and a simplify and consolidate network
#     if not G:
#         G = load_simplified_consolidated_graph(rr, radius=radius, intertypes=intertypes)
#     else:
#         # get a deep copy of G so that changes made to local G are not made to the original G
#         G = deepcopy(G)
#
#     if fuel_type == 'battery':
#         G.graph['scenario'] = dict(railroad=rr, range_mi=D * KM2MI, fuel_type=fuel_type,
#                                    desired_deployment_perc=deployment_perc, reroute=reroute,
#                                    switch_tech=switch_tech,
#                                    max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
#                                    eff_kwh_p_batt=eff_energy_p_tender, scenario_code=scenario_code)
#         # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
#         t0 = time.time()
#         if perc_ods is None or perc_ods == 'X':
#             perc_ods = deployment_perc_lookup_table(df_scenario=df_scenario, deployment_perc=deployment_perc)
#         print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         if select_cycles:
#             # select almost all O-D pairs with non-zero flow (leave out the 20% with the lowest flow values; too many)
#             ods, od_flows = ods_by_perc_ton_mi(G=G, perc_ods=od_flow_perc, CCWS_filename=CCWS_filename,
#                                                od_flows_truncate=True)
#         if not select_cycles:
#             ods, od_flows = ods_by_perc_ton_mi(G=G, perc_ods=perc_ods, CCWS_filename=CCWS_filename)
#         G.graph['framework'] = dict(ods=ods)
#         print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
#         G, H = facility_location(G, D=D, ods=ods, od_flows=od_flows, flow_min=perc_ods, select_cycles=select_cycles,
#                                  budget=budget, max_flow=max_flow, extend_graph=extend_graph, od_flow_perc=od_flow_perc,
#                                  deviation_paths=deviation_paths,
#                                  binary_prog=binary_prog, suppress_output=suppress_output)
#         print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))
#
#         # if no facilities are selected
#         if G.graph['number_facilities'] == 0:
#             return G
#
#         t0 = time.time()
#         # 3. reroute flows and get peak and average ton and locomotive flows for each edge
#         G, H = route_flows(G=G, fuel_type=fuel_type, H=H, D=D, CCWS_filename=CCWS_filename,
#                            time_window=time_window, freq=freq,
#                            reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
#         print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 4. facility sizing based on peak flows and utilization based on average flows
#         # load cost by state dataframe and assign to each node
#         # emissions_p_location = elec_rate_state(G, emissions=True, clean_energy=clean_energy,
#         #                                        clean_elec_prem_dolkwh=clean_energy_cost)  # [gCO2/kWh]
#         # cost_p_location = elec_rate_state(G, clean_energy=clean_energy,
#         #                                   clean_elec_prem_dolkwh=clean_energy_cost)  # in [$/MWh]
#
#         G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=D, emissions_obj=emissions_obj,
#                             suppress_output=suppress_output)
#         print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
#         G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
#         # 5.1. TEA
#         G = tea_battery_all_facilities(G, max_util=max_util, station_type=station_type)
#         # baseline and other dropin fuels (easy factor calculation)
#         G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         # G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         # G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         print('TEA:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 5.2. LCA
#         G = lca_battery(G, clean_energy=clean_energy)
#         # baseline and other dropin fuels (easy factor calculation)
#         G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         # G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         # G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         print('LCA:: %s seconds ---' % round(time.time() - t0, 3))
#
#     elif fuel_type == 'hydrogen':
#         G.graph['scenario'] = dict(railroad=rr, range_mi=D * KM2MI, fuel_type=fuel_type,
#                                    desired_deployment_perc=deployment_perc, reroute=reroute,
#                                    switch_tech=switch_tech,
#                                    max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
#                                    eff_kgh2_p_loc=eff_energy_p_tender, scenario_code=scenario_code)
#         # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
#         t0 = time.time()
#         if perc_ods is None or perc_ods == 'X':
#             perc_ods = deployment_perc_lookup_table(df_scenario=df_scenario, deployment_perc=deployment_perc)
#             # perc_ods = deployment_perc_lookup_table(filename=scenario_filename, deployment_perc=deployment_perc)
#         print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         ods, od_flows = ods_by_perc_ton_mi(G=G, perc_ods=perc_ods, CCWS_filename=CCWS_filename)
#         G.graph['framework'] = dict(ods=ods)
#         print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
#         G, H = facility_location(G, D=D, ods=ods, od_flows=od_flows,
#                                  binary_prog=binary_prog, suppress_output=suppress_output)
#         print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))
#
#         # if no facilities are selected
#         if G.graph['number_facilities'] == 0:
#             return G
#
#         t0 = time.time()
#         # 3. reroute flows and get peak and average ton and locomotive flows for each edge
#         G, H = route_flows(G=G, fuel_type=fuel_type, H=H, D=D, CCWS_filename=CCWS_filename,
#                            time_window=time_window, freq=freq,
#                            reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
#         print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 4. facility sizing based on peak flows and utilization based on average flows
#         G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=D, unit_sizing_obj=True, suppress_output=suppress_output)
#         print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
#         G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
#         # 5.1. TEA
#         G = tea_hydrogen_all_facilities(G, max_util=max_util, station_type=station_type)
#         # baseline and other dropin fuels (easy factor calculation)
#         G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         print('TEA:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 5.2. LCA
#         G = lca_hydrogen(G)
#         # baseline and other dropin fuels (easy factor calculation)
#         G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
#         print('LCA:: %s seconds ---' % round(time.time() - t0, 3))
#
#     elif fuel_type == 'e-fuel' or fuel_type == 'biodiesel' or fuel_type == 'diesel':
#         if deployment_perc is None:
#             deployment_perc = 1
#
#         G.graph['scenario'] = dict(railroad=rr, range_mi=np.nan, fuel_type=fuel_type,
#                                    desired_deployment_perc=deployment_perc, reroute=reroute,
#                                    switch_tech=switch_tech,
#                                    max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
#                                    scenario_code=scenario_code)
#
#         t0 = time.time()
#         # 1. route baseline flows to get average daily ton and locomotive flows for each edge
#         G = route_flows(G=G, fuel_type=fuel_type, CCWS_filename=CCWS_filename, time_window=time_window)
#         print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 2. TEA
#         G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         G = tea_dropin(G, fuel_type=fuel_type, deployment_perc=deployment_perc)
#         print('TEA:: %s seconds ---' % round(time.time() - t0, 3))
#
#         t0 = time.time()
#         # 3. LCA
#         G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
#         G = lca_dropin(G, fuel_type=fuel_type, deployment_perc=deployment_perc)
#         print('LCA:: %s seconds ---' % round(time.time() - t0, 3))
#
#     G = operations_stats(G)
#     print('SCENARIO RUN:: %s seconds ---' % round(time.time() - t0_total, 3))
#
#     return G
