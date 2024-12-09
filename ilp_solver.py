import pandas as pd

from util import *
from solver import Solver
from helper import gurobi_suppress_output


class ILPSolver(Solver):

    def __init__(self, G: nx.DiGraph, range_km: dict, time_horizon: list, od_flows: dict,
                 facility_costs: dict = None, max_flow=False, flow_mins: float = None,
                 budgets: list = None, discount_rates: any = None,
                 deviation_paths=True, fixed_facilities: dict = None, barred_facilities: dict = None,
                 y_warm_start: dict = None, warm_start_strategy: str = None, solution_tol: float = None,
                 nested=True, num_shortest_paths: int = 1, od_flow_perc: float = 1.0,
                 solution_aids=False, solution_time_limit=None,
                 binary_prog=False, suppress_output=True, scenario_code: str = None, solver_param_code: str = None):
        super().__init__(G=G, range_km=range_km, time_horizon=time_horizon, od_flows=od_flows,
                         facility_costs=facility_costs, max_flow=max_flow, flow_mins=flow_mins, budgets=budgets,
                         discount_rates=discount_rates,
                         deviation_paths=deviation_paths, fixed_facilities=fixed_facilities,
                         barred_facilities=barred_facilities, y_warm_start=y_warm_start,
                         warm_start_strategy=warm_start_strategy, solution_tol=solution_tol, nested=nested,
                         num_shortest_paths=num_shortest_paths, od_flow_perc=od_flow_perc,
                         solution_time_limit=solution_time_limit, binary_prog=binary_prog,
                         suppress_output=suppress_output, scenario_code=scenario_code,
                         solver_param_code=solver_param_code)
        self.solution_aids = solution_aids

    def write(self):
        super().write()

    def solve(self):
        # call the correct solver method depending on input parameters (max_flow, deviation_paths, etc)
        # store the returned values in self.y_val, self.z_val, etc.
        super().pre_process_parameters()

        # TODO: include any additional/alternative solution methods as needed (min cost, p2p, etc.) and divert as needed
        if self.deviation_paths:
            if self.max_flow:
                self.solve_mp_dc_frlm_max()

    def solve_mp_dc_frlm_max(self):
        # implementation of mp_dc_frlm_max_ilp, return necessary values for storage in G

        # set up model

        m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(self.suppress_output))
        # facility location and flow selection variables
        if self.binary_prog:
            y = {t: m.addMVar((self.n,), vtype=GRB.BINARY, lb=self.y_lb[t], ub=self.y_ub[t],
                              name='y_{v1}'.format(v1=t)) for t in self.time_horizon}
            z = {t: m.addMVar((len(self.pod_comb),), vtype=GRB.BINARY, name='z_{v1}'.format(v1=t))
                 for t in self.time_horizon}
            w = {t: m.addMVar((len(self.pdo_comb),), vtype=GRB.BINARY, name='w_{v1}'.format(v1=t))
                 for t in self.time_horizon}
        else:
            y = {t: m.addMVar((self.n,), vtype=GRB.BINARY, lb=self.y_lb[t], ub=self.y_ub[t],
                              name='y_{v1}'.format(v1=t)) for t in self.time_horizon}
            z = {t: m.addMVar((len(self.pod_comb),), lb=0, name='z_{v1}'.format(v1=t))
                 for t in self.time_horizon}
            w = {t: m.addMVar((len(self.pdo_comb),), lb=0, name='w_{v1}'.format(v1=t))
                 for t in self.time_horizon}

        # initialize problem with warm start solution
        if self.y_ws:
            for t in self.time_horizon:
                y[t].setAttr('Start', self.y_ws[t])

        # objective fxn
        # m.setObjective(gp.quicksum(discount_rates[t] * pod_flows[t].transpose() @ z[t] for t in time_horizon), GRB.MAXIMIZE)
        # this objective function is minorly tweaked to pick the lowest cost solution among multiple optimal solutions
        # by (a) selecting the optimal solution with the fewest facilities and
        # (b) routing service cycles on the shortest paths available
        m.setObjective(gp.quicksum(self.discount_rates[t] * self.pod_flows[t].transpose() @ z[t]
                                   for t in self.time_horizon)
                       # - gp.quicksum(1e-6 * np.ones((self.n,)).transpose() @ y[t] for t in self.time_horizon)
                       # - gp.quicksum(1e-6 * dev_pod_lengths.transpose() @ z[t] for t in time_horizon)
                       # - gp.quicksum(1e-6 * dev_pdo_lengths.transpose() @ w[t] for t in time_horizon)
                       , GRB.MAXIMIZE)

        # constraints
        for t_idx, t in enumerate(self.time_horizon):
            # outbound path facility-flow coverage
            m.addConstr(self.A_out[self.range_km[t]] @ y[t] >= self.H_out @ z[t], name='C2_{v1}'.format(v1=t))
            # return path facility-flow coverage
            m.addConstr(self.A_rev[self.range_km[t]] @ y[t] >= self.H_rev @ w[t], name='C3_{v1}'.format(v1=t))
            # outbound-return path-od relation
            m.addConstr(self.E_out @ z[t] <= self.E_rev @ w[t], name='C4_{v1}'.format(v1=t))
            # outbound path-od relation (redundant - can remove)
            m.addConstr(self.E_out @ z[t] <= np.ones((self.E_out.shape[0],)), name='C5_{v1}'.format(v1=t))
            # return path-od relation
            m.addConstr(self.E_rev @ w[t] <= np.ones((self.E_rev.shape[0],)), name='C6_{v1}'.format(v1=t))

            # budget constraints
            if t_idx == 0:
                m.addConstr(self.facility_costs[t] @ y[t] <= self.budgets[t], name='C7_{v1}'.format(v1=t))
            else:
                m.addConstr(self.facility_costs[t] @ y[t] - self.facility_costs[t] @ y[self.time_horizon[t_idx - 1]]
                            <= self.budgets[t], name='C7_{v1}'.format(v1=t))

            # facility nestedness constraints
            if self.nested:
                if t != self.time_horizon[-1]:
                    m.addConstr(y[t] <= y[self.time_horizon[t_idx + 1]], name='C8_{v1}'.format(v1=t))

        # remove solution aids from gurobi
        if not self.solution_aids:
            m.setParam('Presolve', 0)  # turn off presolve
            m.setParam('Aggregate', 0)  # turn off aggregation
            m.setParam('Cuts', 0)
            m.setParam('Heuristics', 0)
            m.setParam('Symmetry', 0)
            m.setParam('Disconnected', 0)
            m.setParam('Method', 0)  # primal simplex only (for continuous relaxations of the model)

        # set solution time limit
        if self.solution_time_limit is not None:
            m.setParam('TimeLimit', self.solution_time_limit)  # set solution time limit

        # set solution tolerance
        if self.solution_tol:
            m.setParam('MIPGap', self.solution_tol)

        # log ILP model
        # file_name = f'mp_dc_frlm_max_{time.time()}.csv'
        # m.setParam('LogFile', os.path.join(GRB_MODEL_DIR, file_name))
        # m.write('/Users/adrianhz/Desktop/KCS_test_DP_ILP.lp')
        # m.write('/Users/adrianhz/Desktop/KCS_test_DP_ILP.rlp')    # to be able to be read back by gurobi

        t1 = time.time()
        # optimize
        m.update()
        m.optimize()
        self.run_time += time.time() - t1
        self.gap = m.MIPGap
        self.obj_val = m.objval

        # if time limit not exceeded
        if m.Status != 9:
            # extract solution values
            self.y_val = {t: y[t].X for t in self.time_horizon}
            self.z_val = {t: z[t].X for t in self.time_horizon}
            self.w_val = {t: w[t].X for t in self.time_horizon}
            # self.obj_val = m.objval
        else:
            self.time_limit_reached = True
