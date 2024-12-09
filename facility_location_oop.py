from util import *
from solver import Solver
from ilp_solver import ILPSolver
from greedy_solver import GreedySolver
from benders_solver import BendersSolver
from helper import covered_graph


# TODO: rename methods to match problem variant;
#  - combine all static+dynamic ILP writing and solution methods in one module
#  - combine all matrix/adjacent methods in a separate module
# TODO: separate facility rollout from routing, sizing, tea, lca, and plotting components here
#  create a separate repository? or index all relevant files by '_mp' for multi-period?

# LOCATION


# TODO: to split Solver, ILPSolver, and GreedySolver into their own modules later


def facility_location_mp(G: nx.DiGraph, range_km: dict, time_horizon: list, od_flows: dict,
                         facility_costs: dict = None,
                         max_flow=False, greedy=False, flow_mins: float = None, budgets: list = None,
                         discount_rates: any = None, deviation_paths=True,
                         fixed_facilities: dict = None, barred_facilities: dict = None,
                         y_warm_start: dict = None, warm_start_strategy: str = None,
                         solution_tol: float = None,
                         nested=True, num_shortest_paths: int = 1, od_flow_perc: float = 1.0,
                         benders=False, strong_cuts=False, multi_cut_time=True, multi_cut_ods=True,
                         analytic_sp=True, max_iter=200,
                         solution_aids=False, solution_time_limit=None,
                         binary_prog=False, suppress_output=True, print_status=True,
                         scenario_code: str = None, solver_param_code: str = None):
    if greedy:
        s = GreedySolver(G=G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                         facility_costs=facility_costs, max_flow=max_flow, flow_mins=flow_mins, budgets=budgets,
                         discount_rates=discount_rates,
                         deviation_paths=deviation_paths, fixed_facilities=fixed_facilities,
                         barred_facilities=barred_facilities, y_warm_start=y_warm_start,
                         warm_start_strategy=warm_start_strategy, solution_tol=solution_tol, nested=nested,
                         num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                         solution_time_limit=solution_time_limit, binary_prog=binary_prog,
                         suppress_output=suppress_output, scenario_code=scenario_code,
                         solver_param_code=solver_param_code)
    else:
        if benders:
            s = BendersSolver(G=G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                              facility_costs=facility_costs, max_flow=max_flow, flow_mins=flow_mins, budgets=budgets,
                              discount_rates=discount_rates,
                              deviation_paths=deviation_paths, fixed_facilities=fixed_facilities,
                              barred_facilities=barred_facilities, y_warm_start=y_warm_start,
                              warm_start_strategy=warm_start_strategy, solution_tol=solution_tol, nested=nested,
                              num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                              strong_cuts=strong_cuts, multi_cut_time=multi_cut_time, multi_cut_ods=multi_cut_ods,
                              analytic_sp=analytic_sp, max_iter=max_iter,
                              solution_time_limit=solution_time_limit, binary_prog=binary_prog,
                              suppress_output=suppress_output, print_status=print_status,
                              scenario_code=scenario_code, solver_param_code=solver_param_code)
        else:
            s = ILPSolver(G=G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                          facility_costs=facility_costs, max_flow=max_flow, flow_mins=flow_mins, budgets=budgets,
                          discount_rates=discount_rates,
                          deviation_paths=deviation_paths, fixed_facilities=fixed_facilities,
                          barred_facilities=barred_facilities, y_warm_start=y_warm_start,
                          warm_start_strategy=warm_start_strategy, solution_tol=solution_tol, nested=nested,
                          num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                          solution_aids=solution_aids, solution_time_limit=solution_time_limit,
                          binary_prog=binary_prog, suppress_output=suppress_output, scenario_code=scenario_code,
                          solver_param_code=solver_param_code)

    s.solve()
    s.write()

    return s.process_solution(), s.time_limit_reached
