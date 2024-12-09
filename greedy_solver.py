from util import *
from solver import Solver
from helper import gurobi_suppress_output

class GreedySolver(Solver):

    def __init__(self, G: nx.DiGraph, range_km: dict, time_horizon: list, od_flows: dict,
                 facility_costs: dict = None, max_flow=False, flow_mins: float = None,
                 budgets: list = None, discount_rates: any = None,
                 deviation_paths=True, fixed_facilities: dict = None, barred_facilities: dict = None,
                 y_warm_start: dict = None, warm_start_strategy: str = None, solution_tol: float = None,
                 nested=True, num_shortest_paths: int = 1, od_flow_perc: float = 1.0, solution_time_limit=None,
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

    def write(self):
        super().write()

    def solve(self):
        # call the correct solver method depending on input parameters (max_flow, deviation_paths, etc)
        # store the returned values in self.y_val, self.z_val, etc.
        super().pre_process_parameters()

        if self.deviation_paths:
            if self.max_flow:
                self.solve_mp_dc_frlm_max_greedy()

    def solve_mp_dc_frlm_max_greedy(self):
        # set up and run backwards myopic model (based on Chung and Kwon, 2015)
        cum_budgets = {t_step: sum(self.budgets[t] for t in self.time_horizon[:t_idx + 1])
                       for t_idx, t_step in enumerate(self.time_horizon)}
        self.y_val = {t: [] for t in self.time_horizon}
        self.z_val = {t: [] for t in self.time_horizon}
        self.w_val = {t: [] for t in self.time_horizon}
        self.obj_val = 0
        t_future = -1
        iter_count = 1
        for t in list(reversed(self.time_horizon)):
            # set up model
            m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(self.suppress_output))
            # facility upper bounds (based on fixed facilities)
            if t_future == -1:
                ub = np.ones((self.n,))
            else:
                ub = self.y_val[t_future]
            # facility location and flow selection variables
            y = m.addMVar((self.n,), vtype=GRB.BINARY, ub=ub, name='y_{v1}'.format(v1=t))
            z = m.addMVar((len(self.pod_comb),), vtype=GRB.BINARY, name='z_{v1}'.format(v1=t))
            w = m.addMVar((len(self.pdo_comb),), vtype=GRB.BINARY, name='w_{v1}'.format(v1=t))

            # objective fxn
            m.setObjective(self.pod_flows[t].transpose() @ z, GRB.MAXIMIZE)

            # outbound path facility-flow coverage
            m.addConstr(self.A_out[self.range_km[t]] @ y >= self.H_out @ z, name='C2_{v1}'.format(v1=t))
            # return path facility-flow coverage
            m.addConstr(self.A_rev[self.range_km[t]] @ y >= self.H_rev @ w, name='C3_{v1}'.format(v1=t))
            # outbound-return path-od relation
            m.addConstr(self.E_out @ z <= self.E_rev @ w, name='C4_{v1}'.format(v1=t))
            # outbound path-od relation (redundant - can remove)
            # m.addConstr(E_out @ z[t] <= np.ones((E_out.shape[0],)), name='C5_{v1}'.format(v1=t))
            # return path-od relation
            m.addConstr(self.E_rev @ w <= np.ones((self.E_rev.shape[0],)), name='C6_{v1}'.format(v1=t))
            # budget constraints (cumulative here)
            m.addConstr(self.facility_costs[t] @ y <= cum_budgets[t], name='C7_{v1}'.format(v1=t))

            t1 = time.time()
            # optimize
            m.update()
            m.optimize()
            self.run_time += time.time() - t1

            # extract solution values
            self.y_val[t] = y.X
            self.z_val[t] = z.X
            self.w_val[t] = w.X
            self.obj_val += self.discount_rates[t] * m.ObjVal  # get objective fxn value

            # update for next step
            t_future = t
            iter_count += 1
