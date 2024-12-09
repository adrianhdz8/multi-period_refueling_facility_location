from util import *
from helper import input_cleaning, gurobi_suppress_output, node_to_edge_path, covered_graph_time
from matrix_constructor import comb_constraint_matrix_k_time_dc_mp


class Solver():

    def __init__(self, G: nx.DiGraph, range_km: dict, time_horizon: list, od_flows: dict,
                 facility_costs: dict = None, max_flow=False, flow_mins: float = None,
                 budgets: list = None, discount_rates: any = None,
                 deviation_paths=True, fixed_facilities: dict = None, barred_facilities: dict = None,
                 y_warm_start: dict = None, warm_start_strategy: str = None, solution_tol: float = None,
                 nested=True, num_shortest_paths: int = 1, od_flow_perc: float = 1.0, solution_time_limit=None,
                 binary_prog=False, suppress_output=True, scenario_code: str = None, solver_param_code: str = None):
        # file codes
        self.scenario_code = scenario_code
        self.solver_param_code = solver_param_code
        # scenario params
        self.G = G
        self.time_horizon = time_horizon
        self.range_km = range_km
        self.num_shortest_paths = num_shortest_paths
        self.od_flow_perc = od_flow_perc
        self.od_flows = od_flows
        # formulation set and coefficient params
        self.discount_rates = discount_rates
        self.facility_costs = facility_costs
        self.flow_mins = flow_mins
        self.budgets = budgets
        self.fixed_facilities = fixed_facilities
        self.barred_facilities = barred_facilities
        # formulation type params
        self.max_flow = max_flow
        self.deviation_paths = deviation_paths
        self.nested = nested
        # warm start params
        self.y_warm_start = y_warm_start
        self.warm_start_strategy = warm_start_strategy
        # solver params
        self.solution_tol = solution_tol
        self.solution_time_limit = solution_time_limit
        self.binary_prog = binary_prog
        self.suppress_output = suppress_output
        # solution values
        self.y_val = None
        self.z_val = None
        self.w_val = None
        self.obj_val = None
        self.run_time = 0
        self.gap = np.inf
        self.time_limit_reached = False

    def pre_process_parameters(self):
        # pre-process all parameters and ensure they are parsed correctly
        # pre-compute matrices needed by solvers (use matrix_constructor.py methods)
        # any warm-starting, etc.
        # store all information in corresponding <self.param> items
        # should be universal to all solution methods

        # process input parameters and ensure all are valid
        [self.od_flows, self.budgets,
         self.facility_costs, self.discount_rates] = input_cleaning(G=self.G, time_horizon=self.time_horizon,
                                                                    od_flows=self.od_flows,
                                                                    budgets=self.budgets, flow_mins=self.flow_mins,
                                                                    facility_costs=self.facility_costs,
                                                                    discount_rates=self.discount_rates)
        # construct matrices needed for writing and solving the program instances
        ([self.A_out, self.H_out, self.npod_comb, self.pod_comb,
          self.A_rev, self.H_rev, self.npdo_comb, self.pdo_comb, self.node_list],
         self.deviation_paths) = comb_constraint_matrix_k_time_dc_mp(G=self.G, ods=list(self.od_flows.keys()),
                                                                     node_list=list(self.G.nodes),
                                                                     range_km_set=set(self.range_km.values()),
                                                                     num_shortest_paths=self.num_shortest_paths,
                                                                     od_flow_perc=self.od_flow_perc)
        # ([self.A_out, self.H_out, self.npod_comb, self.pod_comb,
        #   self.A_rev, self.H_rev, self.npdo_comb, self.pdo_comb, self.node_list],
        #  self.deviation_paths) = comb_constraint_matrix_k_dc_mp(G=self.G, ods=list(self.od_flows.keys()),
        #                                                         node_list=list(self.G.nodes),
        #                                                         range_km=self.range_km,
        #                                                         num_shortest_paths=self.num_shortest_paths,
        #                                                         od_flow_perc=self.od_flow_perc)

        self.n = len(self.node_list)
        self.od_list = list({od for _, od in self.pod_comb})
        od_idx_dict = {v: i for i, v in enumerate(self.od_list)}
        self.do_list = [(d, o) for o, d in self.od_list]
        do_idx_dict = {v: i for i, v in enumerate(self.do_list)}
        od_flows_arrays = {t: np.zeros((len(self.od_list), 1)) for t in self.time_horizon}
        for t in self.time_horizon:
            for i, (o, d) in enumerate(self.od_list):
                od_flows_arrays[t][i] += self.od_flows[o, d][t] if (o, d) in self.od_flows.keys() else 0
                od_flows_arrays[t][i] += self.od_flows[d, o][t] if (d, o) in self.od_flows.keys() else 0
            # rescale values
            od_flows_arrays[t] = od_flows_arrays[t] / 1e5

        # sparse matrix with [p, od] in rows and [od] in cols = 1 if <p> in set of deviation paths for <od>, 0 o.w.
        # row_idxs can be used to retreive combinations from pod_comb list
        # <E_out_T> is a ( len(pod_comb) x len(od_list) ) 0-1 matrix
        self.E_out = csr_matrix((np.ones((len(self.pod_comb),)),
                                 (np.array([i for i in range(len(self.pod_comb))]),
                                  np.array([od_idx_dict[od] for _, od in self.pod_comb]))),
                                shape=(len(self.pod_comb), len(self.od_list))).T
        self.E_rev = csr_matrix((np.ones((len(self.pdo_comb),)),
                                 (np.array([i for i in range(len(self.pdo_comb))]),
                                  np.array([do_idx_dict[do] for _, do in self.pdo_comb]))),
                                shape=(len(self.pdo_comb), len(self.do_list))).T

        self.pod_flows = {t: self.E_out.T @ od_flows_arrays[t] for t in self.time_horizon}

        self.y_ws, self.y_lb, self.y_ub = self.mp_dc_frlm_max_greedy_initialization()

        # percentage gap
        if self.solution_tol is None:
            self.solution_tol = 0.01

    def mp_dc_frlm_max_greedy_initialization(self):

        y_ws = None
        y_lb = {t: np.zeros((self.n,)) for t in self.time_horizon}
        y_ub = {t: np.ones((self.n,)) for t in self.time_horizon}

        if self.warm_start_strategy == 'greedy' or self.fixed_facilities == 'static':
            y_greedy = self.y_warm_start_greedy()
            if self.warm_start_strategy == 'greedy':
                y_ws = y_greedy

            if self.fixed_facilities == 'static':
                y_lb[self.time_horizon[-1]] = np.array([1 if y_greedy[self.time_horizon[-1]][j] == 1
                                                        else 0 for j in range(self.n)])

        for t in self.time_horizon:
            if isinstance(self.fixed_facilities, dict):
                if t in self.fixed_facilities.keys():
                    y_lb[t] = np.array([1 if j in self.fixed_facilities[t] else 0 for j in self.node_list])
            if isinstance(self.barred_facilities, dict):
                if t in self.barred_facilities.keys():
                    y_ub[t] = np.array([0 if j in self.barred_facilities[t] else 1 for j in self.node_list])

        return y_ws, y_lb, y_ub

    def y_warm_start_greedy(self):
        # TODO: use <GreedySolver> for this instead?

        # set up and run backwards myopic model (based on Chung and Kwon, 2015)
        cum_budgets = {t_step: sum(self.budgets[t] for t in self.time_horizon[:t_idx + 1])
                       for t_idx, t_step in enumerate(self.time_horizon)}
        y_val = {t: [] for t in self.time_horizon}

        t_future = -1
        iter_count = 1
        for t in list(reversed(self.time_horizon)):
            # set up model
            m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(self.suppress_output))
            # facility upper bounds (based on fixed facilities)
            if t_future == -1:
                ub = np.ones((self.n,))
            else:
                ub = y_val[t_future]
            # facility location and flow selection variables
            y = m.addMVar((self.n,), vtype=GRB.BINARY, ub=ub, name='y_{v1}'.format(v1=t))
            z = m.addMVar((len(self.pod_comb),), lb=0, ub=1, name='z_{v1}'.format(v1=t))
            w = m.addMVar((len(self.pdo_comb),), lb=0, ub=1, name='w_{v1}'.format(v1=t))

            # objective fxn
            m.setObjective(self.pod_flows[t].transpose() @ z, GRB.MAXIMIZE)

            # constraints
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

            # optimize
            m.update()
            m.optimize()

            # extract solution values
            y_val[t] = np.abs(y.X)

            # update for next step
            t_future = t
            iter_count += 1

        return y_val

    def process_solution(self):

        if self.time_limit_reached:
            return self.G

        # store all solution and relevant information in G dictionary and return G
        od_path_list = {od: [] for od in self.od_list}
        for p, od in self.pod_comb:
            od_path_list[od].append(p)
        do_path_list = {do: [] for do in self.do_list}
        for p, do in self.pdo_comb:
            do_path_list[do].append(p)

        pod_idx_dict = {v: i for i, v in enumerate(self.pod_comb)}
        pdo_idx_dict = {v: i for i, v in enumerate(self.pdo_comb)}
        # rounding scheme for fractional z, w
        for t in self.time_horizon:
            # outbound paths (z)
            for od in od_path_list:
                unit_pod_comb_idx = -1
                zero_pod_comb_idxs = []
                for p in range(len(od_path_list[od])):
                    pod_idx = pod_idx_dict[p, od]
                    # if combination value is activated (fractional or unit) set to unit and all others to zero
                    if self.solution_tol < self.z_val[t][pod_idx] and unit_pod_comb_idx == -1:
                        unit_pod_comb_idx = pod_idx
                    else:
                        zero_pod_comb_idxs.append(pod_idx)
                # if one combination fractional, set to 1
                if unit_pod_comb_idx != -1:
                    self.z_val[t][unit_pod_comb_idx] = 1
                # set the rest to 0
                for pod_idx in zero_pod_comb_idxs:
                    self.z_val[t][pod_idx] = 0
            # return paths (w)
            for do in do_path_list:
                unit_pdo_comb_idx = -1
                zero_pdo_comb_idxs = []
                for p in range(len(do_path_list[do])):
                    pdo_idx = pdo_idx_dict[p, do]
                    # if combination value is activated (fractional or unit) set to unit and all others to zero
                    if self.solution_tol < self.w_val[t][pdo_idx] and unit_pdo_comb_idx == -1:
                        unit_pdo_comb_idx = pdo_idx
                    else:
                        zero_pdo_comb_idxs.append(pdo_idx)
                # if one combination fractional, set to 1
                if unit_pdo_comb_idx != -1:
                    self.w_val[t][unit_pdo_comb_idx] = 1
                # set the rest to zero
                for pdo_idx in zero_pdo_comb_idxs:
                    self.w_val[t][pdo_idx] = 0

        phi_list = {od: [(paths_idx, p_idx) for paths_idx, paths in enumerate(self.deviation_paths[od])
                         for p_idx, _ in enumerate(paths)] for od in self.deviation_paths.keys()}

        covered_path_nodes = {t: {od: set() for od in self.od_list} for t in self.time_horizon}
        covered_path_edges = {t: {od: set() for od in self.od_list} for t in self.time_horizon}
        for t in self.time_horizon:
            for pod_idx, (p, (o, d)) in enumerate(self.pod_comb):
                if self.z_val[t][pod_idx] == 1:
                    phi_1_idx, phi_2_idx = phi_list[o, d][p]
                    covered_path_nodes[t][o, d].update(self.deviation_paths[o, d][phi_1_idx][phi_2_idx])
                    covered_path_edges[t][o, d].update(
                        node_to_edge_path(self.deviation_paths[o, d][phi_1_idx][phi_2_idx]))
        # # return path
        # covered_path_nodes_ret = {t: {od: set() for od in self.od_list} for t in self.time_horizon}
        # covered_path_edges_ret = {t: {od: set() for od in self.od_list} for t in self.time_horizon}
        # for t in self.time_horizon:
        #     for pdo_idx, (p, (d, o)) in enumerate(self.pdo_comb):
        #         if self.w_val[t][pdo_idx] == 1:
        #             phi_1_idx, phi_2_idx = phi_list[o, d][p]
        #             covered_path_nodes_ret[t][o, d].update(self.deviation_paths[o, d][phi_1_idx][phi_2_idx])
        #             covered_path_edges_ret[t][o, d].update(
        #                 node_to_edge_path(self.deviation_paths[o, d][phi_1_idx][phi_2_idx]))

        self.G.graph['framework'] = dict(
            time_horizon=self.time_horizon,
            facility_costs=self.facility_costs,
            candidate_facilities=None,
            discount_rates=self.discount_rates,
            budgets=self.budgets,
            cum_budgets={t_step: sum(self.budgets[t] for t in self.time_horizon[:t_idx + 1])
                         for t_idx, t_step in enumerate(self.time_horizon)},
            od_deviation_path_flows=self.pod_flows,
            ods=self.od_list,
            selected_ods={t: list(set(od for pod_idx, (p, od) in enumerate(self.pod_comb)
                                      if self.z_val[t][pod_idx] == 1)) for t in self.time_horizon},
            selected_facilities={t: set(self.node_list[j] for j in range(self.n) if self.y_val[t][j] == 1)
                                 for t in self.time_horizon},
            covered_path_nodes=covered_path_nodes,
            covered_path_edges=covered_path_edges,
            # covered_path_nodes_ret=covered_path_nodes_ret,
            # covered_path_edges_ret=covered_path_edges_ret,
            tm_available={t: 1e5 * sum(self.od_flows[od][t] for od in self.od_list) for t in self.time_horizon},
            tm_capt={t: 1e5 * self.pod_flows[t].transpose() @ self.z_val[t] for t in self.time_horizon},
            tm_capt_perc={t: (1e5 * self.pod_flows[t].transpose() @ self.z_val[t] /
                              sum(self.od_flows[od][t] for od in self.od_list)) for t in self.time_horizon},
            # these are cumulative discounted flow capture
            tm_capt_final=1e5 * self.obj_val,
            tm_capt_perc_final=1e5 * self.obj_val / sum(
                self.od_flows[od][self.time_horizon[-1]] for od in self.od_list),
            obj_val=1e5 * self.obj_val,
            od_flows=self.od_flows,
            y_val=self.y_val,
            z_val=self.z_val,
            w_val=self.w_val,
            run_time=self.run_time,
            deviation_paths=self.deviation_paths,
            npod_comb=self.npod_comb,
            npdo_comb=self.npdo_comb,
            pod_comb=self.pod_comb,
            pdo_comb=self.pdo_comb,
            pod_flows=self.pod_flows,
            node_list=self.node_list,
        )
        self.G.graph['framework']['tm_capt'] = {t: self.G.graph['framework']['tm_capt'][t][0]
                                                for t in self.time_horizon}
        self.G.graph['framework']['tm_capt_perc'] = {t: self.G.graph['framework']['tm_capt_perc'][t][0]
                                                     for t in self.time_horizon}
        self.G.graph['framework']['cum_tm_capt'] = {t: sum(self.G.graph['framework']['tm_capt'][self.time_horizon[tj]]
                                                           for tj in range(ti + 1))
                                                    for ti, t in enumerate(self.time_horizon)}

        # TODO: fix this to depend on time-dependent range
        return covered_graph_time(G=self.G, range_km=self.range_km, extend_graph=False)

    def write(self):
        # log solution performance for instance (scenario_code) and solver info (solver_param_file)
        #  - final obj value, gap, solution time
        if self.scenario_code is not None and self.solver_param_code is not None:
            pd.DataFrame(data=[[self.gap, self.obj_val, self.run_time, int(self.time_limit_reached),
                                self.num_shortest_paths, self.G.number_of_nodes(), self.G.number_of_edges(),
                                len(self.od_list)]],
                         columns=['GAP', 'OBJ_VAL', 'T(CUM)', 'TIME_LIMIT_REACHED',
                                  '# SHORTEST_PATHS', '# NODES', '# EDGES', '# ODS']).to_csv(
                os.path.join(EXP_DIR, self.scenario_code + '_' + self.solver_param_code + '.csv'))
