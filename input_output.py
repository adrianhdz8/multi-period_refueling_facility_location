import datetime
import os.path

import numpy as np
import pandas as pd

from network_processing import load_simplified_consolidated_graph
from util import *


def load_scenario_mp_df(scenario_code: str):
    filename = scenario_code + '.csv'
    df_scenario = pd.read_csv(os.path.join(SCENARIO_DIR, filename), header=0, index_col='Keyword')
    # convert str values to int/float where possible
    df_scenario.loc['time_horizon'] = df_scenario.loc['time_horizon'].apply(lambda x: str(int(x)))
    df_scenario.loc['flow_mins'] = df_scenario.loc['flow_mins'].apply(lambda x: float(x))
    df_scenario.loc['budgets'] = df_scenario.loc['budgets'].apply(lambda x: float(x))
    df_scenario.loc['range_km'] = df_scenario.loc['range_km'].apply(lambda x: float(x))
    df_scenario.loc['discount_rates'] = df_scenario.loc['discount_rates'].apply(lambda x: float(x))
    int_vals = ['max_flow', 'num_shortest_paths', 'deviation_paths', 'nested', 'constant_flows', 'od_flows_provided',
                'plot', 'colors', 'time_step_label']
    for int_val in int_vals:
        df_scenario.loc[int_val, 'Value0'] = int(df_scenario.loc[int_val, 'Value0'])
    float_vals = ['od_flow_perc']
    for float_val in float_vals:
        df_scenario.loc[float_val, 'Value0'] = float(df_scenario.loc[float_val, 'Value0'])

    return df_scenario


def load_solver_df(solver_param_code: str):
    filename = solver_param_code + '.csv'
    df_scenario = pd.read_csv(os.path.join(SOLVER_PARAM_DIR, filename), header=0, index_col='Keyword')
    num_vals = ['greedy', 'solution_aids', 'benders', 'strong_cuts', 'multi_cut_time', 'multi_cut_ods', 'analytic_sp',
                'max_iter', 'solution_tol', 'solution_time_limit', 'binary_prog', 'suppress_output', 'print_status']
    df_scenario.loc[num_vals, 'Value'] = pd.to_numeric(df_scenario.loc[num_vals, 'Value'], errors='coerce')

    return df_scenario


def extract_assert_param_inputs(solver_param_code: str):

    df_params = load_solver_df(solver_param_code=solver_param_code)
    idxs = ['greedy', 'solution_aids', 'benders', 'strong_cuts', 'multi_cut_time', 'multi_cut_ods', 'analytic_sp',
            'warm_start_strategy', 'max_iter', 'solution_tol', 'solution_time_limit', 'binary_prog', 'suppress_output',
            'print_status']

    [greedy, solution_aids, benders, strong_cuts, multi_cut_time, multi_cut_ods, analytic_sp,
     warm_start_strategy, max_iter, solution_tol, solution_time_limit,
     binary_prog, suppress_output, print_status] = df_params.reindex(idxs)['Value']

    if np.isnan(greedy):
        greedy = False
    if np.isnan(solution_aids):
        solution_aids = False
    if np.isnan(benders):
        benders = False
    if np.isnan(strong_cuts):
        strong_cuts = False
    if np.isnan(multi_cut_time):
        multi_cut_time = False
    if np.isnan(multi_cut_ods):
        multi_cut_ods = False
    if np.isnan(analytic_sp):
        analytic_sp = False
    if not isinstance(warm_start_strategy, str):
        if np.isnan(warm_start_strategy):
            warm_start_strategy = None
    if np.isnan(max_iter):
        max_iter = np.inf
    if np.isnan(solution_tol):
        solution_tol = 1e-4
    if np.isnan(solution_time_limit):
        solution_time_limit = None
    if np.isnan(binary_prog):
        binary_prog = True
    if np.isnan(suppress_output):
        suppress_output = False
    if np.isnan(print_status):
        print_status = True

    return [greedy, solution_aids, benders, strong_cuts, multi_cut_time, multi_cut_ods, analytic_sp,
            warm_start_strategy, max_iter, solution_tol, solution_time_limit,
            binary_prog, suppress_output, print_status]


def write_experiment_summary_file(scenario_codes: list, solver_param_codes: list):
    # scenario_codes = [p[:-4] for p in os.listdir(SCENARIO_DIR) if p[-4:] == '.csv']
    # solver_param_codes = [[p[:-4] for p in os.listdir(SOLVER_PARAM_DIR) if p[-4:] == '.csv']
    #  for _ in range(len(scenario_codes))]

    scenario_code_dict = {s: i for i, s in enumerate(scenario_codes)}
    solver_param_code_dict = {s: i for i, s in enumerate(set([p for q in solver_param_codes for p in q]))}

    gen_cols = ['# SHORTEST_PATHS', '# NODES', '# EDGES', '# ODS']
    gen_multi_cols = [('scenario', i) for i in gen_cols]
    grb_cols = ['GAP', 'OBJ_VAL', 'T(CUM)', 'TIME_LIMIT_REACHED']
    # grb_solver_params = ['sp_gurobi_saon_wsgreedy_0.001_3600_large', 'sp_gurobi_saoff_wsgreedy_0.001_3600_large',
    #                      'sp_gurobi_saon_0.001_3600_large', 'sp_gurobi_saoff_0.001_3600_large', 'sp_greedy']
    grb_solver_params = [s for s in solver_param_code_dict.keys() if s[3:5] != 'bd']

    grb_multi_cols = []
    for s in grb_solver_params:
        grb_multi_cols.extend([(s, i) for i in grb_cols])
    bd_cols =['GAP', 'OBJ_VAL', 'T(CUM)', 'TIME_LIMIT_REACHED', '# CUTS', '# ITER']
    # bd_solver_params = ['sp_bd_mcon_scon_0.001_3600_large', 'sp_bd_mcon_scoff_0.001_3600_large',
    #                      'sp_bd_mcon_scon_wsgreedy_0.001_3600_large', 'sp_bd_mcoff_scoff_0.001_3600_large']
    bd_solver_params = [s for s in solver_param_code_dict.keys() if s[3:5] == 'bd']

    bd_multi_cols = []
    for s in bd_solver_params:
        bd_multi_cols.extend([(s, i) for i in bd_cols])
    all_cols = gen_multi_cols + grb_multi_cols + bd_multi_cols
    multi_cols = pd.MultiIndex.from_tuples(all_cols).sortlevel(level=0)[0]

    df = pd.DataFrame(index=scenario_codes,
                      columns=multi_cols,
                      data=np.nan)

    for sc in scenario_codes:
        for i, s in enumerate(solver_param_code_dict.keys()):
            if os.path.exists(os.path.join(EXP_DIR, sc + '_' + s + '.csv')):
                df_scenario_results = pd.read_csv(os.path.join(EXP_DIR, sc + '_' + s + '.csv'), header=0)
            else:
                continue

            if s in grb_solver_params:
                for col2 in grb_cols:
                    df.loc[sc, (s, col2)] = df_scenario_results.loc[0, col2]
            if s in bd_solver_params:
                for col2 in bd_cols:
                    if col2 != '# ITER':
                        df.loc[sc, (s, col2)] = df_scenario_results.loc[0, col2]
                    else:
                        df.loc[sc, (s, '# ITER')] = df_scenario_results.loc[df_scenario_results.index[-1], 'ITER']
            if i == 0:
                for col2 in gen_cols:
                    df.loc[sc, ('scenario', col2)] = df_scenario_results.loc[0, col2]

    df.to_csv(os.path.join(EXP_DIR, f'all_scenarios_experiments_summary_{datetime.today().day}'
                                    f'{datetime.today().month}{datetime.today().year}.csv'), index=True)

    return df


def load_facility_info(facility_info_filename: str, rr: str, time_horizon):

    G = load_simplified_consolidated_graph(rr=rr)
    node_list = list(G.nodes)
    facility_costs = {(n, t): 1 for t in time_horizon for n in node_list}
    fixed_facilities = {t: [] for t in time_horizon}
    barred_facilities = {t: [] for t in time_horizon}

    if facility_info_filename and os.path.exists(os.path.join(FACILITY_DIR, facility_info_filename)):
        df = pd.read_csv(os.path.join(FACILITY_DIR, facility_info_filename), header=0, index_col='nodeid')
        assert len(df.columns) / 3 == len(time_horizon), \
            'Provide the correct number of time-dependent columns in <facility_info_filename>.'

        for ti, t in enumerate(time_horizon):
            tidx = str(ti)
            for n in df.index:
                if n not in node_list:
                    continue
                facility_costs[(n, t)] = float(df.loc[n, 'cost' + tidx])
                if int(df.loc[n, 'fixed' + tidx]) == 1:
                    fixed_facilities[t].append(n)
                if int(df.loc[n, 'barred' + tidx]) == 1:
                    barred_facilities[t].append(n)
                assert not (int(df.loc[n, 'fixed' + tidx]) == 1 and int(df.loc[n, 'barred' + tidx]) == 1), \
                    'A single facility {n} cannot be both fixed and barred at the same time step {t}.'.format(n=n, t=ti)

    if facility_info_filename == 'static':
        fixed_facilities = 'static'

    return facility_costs, fixed_facilities, barred_facilities


def extract_assert_scenario_mp_inputs(scenario_code: str):

    idxs = ['rr', 'time_horizon', 'range_km', 'budgets', 'max_flow', 'flow_mins', 'discount_rates',
            'num_shortest_paths', 'facility_info_filename',
            'od_flow_perc', 'deviation_paths', 'nested', 'constant_flows', 'od_flows_provided',
            'title', 'plot', 'colors', 'time_step_label']

    df_scenario = load_scenario_mp_df(scenario_code=scenario_code)

    time_inputs = list(df_scenario.columns)
    for i, c in enumerate(time_inputs):
        assert c == 'Value' + str(i), 'Provide valid column names for time-dependent inputs. ' \
                                      'Must be in format of [<Value0>, <Value1>, ..., <ValueT>].'

    [rr, _, _, _, max_flow, _, _, num_shortest_paths, facility_info_filename, od_flow_perc,
     deviation_paths, nested, constant_flows, od_flows_provided,
     title, plot, colors, time_step_label] = df_scenario.reindex(idxs)['Value0']

    # verify validity of inputs
    # rr: railroad selection
    assert rr in {'BNSF', 'UP', 'NS', 'CSXT', 'CN', 'CP', 'USA1', 'WCAN', 'EAST'} or 'KCS' in rr, \
        'Provide a railroad selection <rr>.'
    # time_horizon
    time_horizon = list(df_scenario.loc['time_horizon', time_inputs])
    for ti, t in enumerate(time_horizon):
        assert isinstance(t, int) or isinstance(t, float) or isinstance(t, str), \
            'Provide a valid input type for <time_horizon>.'
        if isinstance(t, int) or isinstance(t, float):
            time_horizon[ti] = str(int(t))
    # range_km
    range_km = list(df_scenario.loc['range_km', time_inputs])
    for ri, r in enumerate(range_km):
        assert isinstance(r, float) or isinstance(r, int), \
            'Provide a valid input for the <range_km> facility location parameter.'
        if ri == 0:
            assert not np.isnan(r), 'At least one <range_km> value must be provided.'
        else:
            if np.isnan(r):
                range_km[ri] = range_km[ri - 1]
    range_km = {t: range_km[ti] for ti, t in enumerate(time_horizon)}
    # max_flow
    assert isinstance(max_flow, bool) or max_flow in {0, 1}, \
        'Provide a valid input for the <max_flow> facility location parameter.'
    max_flow = bool(max_flow)
    # flow_mins
    flow_mins = list(df_scenario.loc['flow_mins', time_inputs])
    for di, d in enumerate(flow_mins):
        assert isinstance(d, float) or isinstance(d, int) and 0 <= d <= 1, \
            'Provide a valid input for <flow_mins>.'
        if np.isnan(d):
            if di != 0:
                flow_mins[di] = flow_mins[di - 1]
            else:
                flow_mins[di] = 0
    assert flow_mins == sorted(flow_mins), 'Provide a valid ordering of <flow_mins>.'
    if not max_flow:
        assert flow_mins[-1] > 0, 'Provide a non-zero value for final input of <flow_mins>.'
    flow_mins = {t: flow_mins[ti] for ti, t in enumerate(time_horizon)}
    # budget
    budgets = list(df_scenario.loc['budgets', time_inputs])
    for bi, b in enumerate(budgets):
        assert isinstance(b, float) or isinstance(b, int), \
            'Provide a valid input for the <budget> facility location parameter.'
        if np.isnan(b):
            if bi != 0:
                budgets[bi] = budgets[bi - 1]
            else:
                budgets[bi] = None
    if max_flow:
        assert not all([b is None for b in budgets]), 'Provide a non-empty value for inputs of <budgets>.'
        assert budgets[-1] > 0, 'Provide a non-zero value for final input of <budgets>.'
    assert budgets == sorted(budgets), 'Provide a valid ordering of <budgets>.'
    budgets = {t: budgets[ti] for ti, t in enumerate(time_horizon)}
    # discount_rates
    discount_rates = list(df_scenario.loc['discount_rates', time_inputs])
    for di, d in enumerate(discount_rates):
        assert isinstance(d, float) or isinstance(d, int) and 0 <= d, \
            'Provide a valid input for <discount_rates>.'
        if np.isnan(d):
            if di != 0:
                discount_rates[di] = discount_rates[di - 1]
            else:
                discount_rates[di] = 1
    assert not all(d == 0 for d in discount_rates), \
        'Provide a non-zero value for at least one input of <discount_rates>.'
    discount_rates = {t: discount_rates[ti] for ti, t in enumerate(time_horizon)}
    # facility_info_filename
    assert isinstance(facility_info_filename, str) or np.isnan(facility_info_filename), \
        'Provide a valid input for <facility_file_name>.'
    if not isinstance(facility_info_filename, str):
        facility_info_filename = None
    facility_costs, fixed_facilities, barred_facilities = load_facility_info(facility_info_filename=
                                                                             facility_info_filename, rr=rr,
                                                                             time_horizon=time_horizon)
    # num_shortest_paths
    assert isinstance(num_shortest_paths, int) and num_shortest_paths > 0, \
        'Provide a valid for <num_shortest_paths> parameter.'
    # od_flow_perc
    assert isinstance(od_flow_perc, int) or isinstance(od_flow_perc, float), \
        'Provide a valid for <od_flow_perc> parameter.'
    assert 0 < od_flow_perc <= 1, \
        'Provide a valid for <od_flow_perc> parameter.'
    # deviation_paths
    assert isinstance(deviation_paths, bool) or deviation_paths in {0, 1}, \
        'Provide a valid input for <deviation_paths> parameter.'
    deviation_paths = bool(deviation_paths)
    # nested
    assert isinstance(nested, bool) or nested in {0, 1}, \
        'Provide a valid input for <nested> parameter.'
    nested = bool(nested)
    # constant_flows
    assert isinstance(constant_flows, bool) or constant_flows in {0, 1}, \
        'Provide a valid input for <constant_flows> parameter.'
    constant_flows = bool(constant_flows)
    # od_flows_provided
    assert isinstance(od_flows_provided, bool) or od_flows_provided in {0, 1}, \
        'Provide a valid input for <constant_flows> parameter.'
    od_flows_provided = bool(od_flows_provided)
    # od_flows_provided
    assert isinstance(od_flows_provided, bool) or od_flows_provided in {0, 1}, \
        'Provide a valid input for <constant_flows> parameter.'
    od_flows_provided = bool(od_flows_provided)
    # title
    assert isinstance(title, str) or np.isnan(title), \
        'Provide a valid input for <title> parameter.'
    # plot
    assert isinstance(plot, bool) or plot in {0, 1}, \
        'Provide a valid input for <plot> parameter.'
    plot = bool(plot)
    # colors
    assert isinstance(colors, bool) or colors in {0, 1}, \
        'Provide a valid input for <colors> parameter.'
    colors = bool(colors)
    # time_step_label
    assert isinstance(time_step_label, bool) or time_step_label in {0, 1}, \
        'Provide a valid input for <time_step_label> parameter.'
    time_step_label = bool(time_step_label)
    # scenario_code
    assert isinstance(scenario_code, str), \
        'Provide a valid input for the <scenario_code> input and output file naming code.'
    assert os.path.exists(os.path.join(SCENARIO_DIR, scenario_code + '.csv')), \
        'Scenario parameter filename does not match <scenario_code> provided.'

    return [rr, time_horizon, range_km, budgets, max_flow, flow_mins, discount_rates, num_shortest_paths,
            facility_costs, fixed_facilities, barred_facilities,
            od_flow_perc, deviation_paths, nested, constant_flows, od_flows_provided,
            title, plot, colors, time_step_label]