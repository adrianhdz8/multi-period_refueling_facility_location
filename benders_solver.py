from util import *
# MODULES
from facility_location_oop import Solver
from helper import node_to_edge_path, gurobi_suppress_output, input_cleaning, mp_dc_frlm_max_greedy_initialization
from matrix_constructor import comb_constraint_matrix_k_dc_mp


class BendersSolver(Solver):

    def __init__(self, G: nx.DiGraph, range_km: dict, time_horizon: list, od_flows: dict,
                 facility_costs: dict = None, max_flow=False, flow_mins: float = None,
                 budgets: list = None, discount_rates: any = None,
                 deviation_paths=True, fixed_facilities: dict = None, barred_facilities: dict = None,
                 y_warm_start: dict = None, warm_start_strategy: str = None, solution_tol: float = None,
                 nested=True, num_shortest_paths: int = 1, od_flow_perc: float = 1.0,
                 solution_time_limit=None, strong_cuts=False, multi_cut_time=True, multi_cut_ods=True,
                 analytic_sp=True, max_iter=None,
                 binary_prog=False, suppress_output=True, print_status=True,
                 scenario_code: str = None, solver_param_code: str = None):
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
        self.strong_cuts = strong_cuts
        self.multi_cut_time = multi_cut_time
        self.multi_cut_ods = multi_cut_ods
        self.analytic_sp = analytic_sp
        self.max_iter = max_iter

        # additional benders' variables
        self.l_val = None
        self.mu_val = None
        self.v_val = None
        self.r_val = None
        self.eta_val = None
        self.y_ip_val = None
        # lists for storing cuts
        self.l_vals = []
        self.mu_vals = []
        self.v_vals = []
        self.r_vals = []
        # log iteration values
        self.print_status = print_status
        self.logged_iterations = []

    def pre_process_parameters(self):
        super().pre_process_parameters()

        # pre-compute and store useful dict objects
        self.pod_idx_dict = {pod: pod_idx for pod_idx, pod in enumerate(self.pod_comb)}
        self.od_ps_dict = {(o, d): [] for o, d in self.od_list}
        for p, (o, d) in self.pod_comb:
            self.od_ps_dict[o, d].append(p)
        self.pdo_idx_dict = {pdo: pdo_idx for pdo_idx, pdo in enumerate(self.pdo_comb)}
        self.do_ps_dict = {(d, o): [] for o, d in self.od_list}
        for p, (d, o) in self.pdo_comb:
            self.do_ps_dict[d, o].append(p)
        self.npod_idx_dict = {npod: npod_idx for npod_idx, npod in enumerate(self.npod_comb)}
        self.odp_npods_dict = {(o, d): {p: [] for p in self.od_ps_dict[o, d]} for (o, d) in self.od_list}
        for (n, p, r, (o, d)) in self.npod_comb:
            self.odp_npods_dict[o, d][p].append((n, p, r, (o, d)))
        self.npdo_idx_dict = {npdo: npdo_idx for npdo_idx, npdo in enumerate(self.npdo_comb)}
        self.dop_npdos_dict = {(d, o): {p: [] for p in self.do_ps_dict[d, o]} for (o, d) in self.od_list}
        for (n, p, r, (d, o)) in self.npdo_comb:
            self.dop_npdos_dict[d, o][p].append((n, p, r, (d, o)))
        self.od_npod_idxs = {od: [] for od in self.od_list}
        for npod_idx, (_, _, _, (o, d)) in enumerate(self.npod_comb):
            self.od_npod_idxs[(o, d)].append(npod_idx)
        self.od_npdo_idxs = {od: [] for od in self.od_list}
        for npdo_idx, (_, _, _, (d, o)) in enumerate(self.npdo_comb):
            self.od_npdo_idxs[(o, d)].append(npdo_idx)

    def initialize_mp_dc_frlm_max_benders(self):

        # warm start initialization
        # TODO: check if necessary/how to simplify
        if self.y_ws is None:
            self.y_ws = {t: np.zeros((self.n,)) for t in self.time_horizon}
        if self.y_warm_start is not None:
            self.y_ws = self.y_warm_start

        self.y_val = self.y_ws

        # interior point initialization
        if sum(self.budgets[t] for t in self.time_horizon) < self.n:
            self.y_ip_val = {t: (sum(self.budgets[self.time_horizon[t_idx2]]
                                     for t_idx2 in range(t_idx + 1)) / self.n) * np.ones((self.n,))
                             for t_idx, t in enumerate(self.time_horizon)}
        else:
            self.y_ip_val = {t: (1 / self.n) * np.ones((self.n,)) for t_idx, t in enumerate(self.time_horizon)}

        self.obj_mp = np.inf
        self.obj_dsp = -np.inf
        self.obj_psp = -np.inf
        self.time_limit_reached = False
        self.gap = np.nan

        self.model_mp = None
        self.y_model_mp = None
        self.eta_model_mp = None
        self.run_time_mp = 0
        self.run_time_dsp = 0
        self.run_time_psp = 0
        self.iteration = 0
        self.number_cuts = 0

        if self.max_iter is None:
            self.max_iter = np.inf

    def log_iteration(self):
        if self.iteration != 0:
            # cols = ['ITER', 'V(MP)', 'V(DSP)', 'GAP', 'OBJ_VAL',
            #         'T(MP)', 'T(DSP)', 'T(TOT)', 'T(CUM)', '# CUTS', 'TIME_LIMIT_REACHED',
            #         '# SHORTEST_PATHS', '# NODES', '# EDGES', '# ODS']
            self.logged_iterations.append([self.iteration, self.obj_mp, self.obj_dsp, self.gap, self.obj_val,
                                           self.run_time_mp, self.run_time_dsp, (self.run_time_mp + self.run_time_dsp),
                                           self.run_time, self.number_cuts, int(self.time_limit_reached),
                                           None, None, None, None])

    def print_iteration_status(self):
        if not self.print_status:
            return

        if self.iteration == 0:
            print('{:<4}  {:>15}  {:>15}  {:>10}  {:>9}  {:>7}  {:>7}  {:>7}  {:>10}'.format('ITER', 'V(MP)',
                                                                                             'V(DSP)', 'GAP',
                                                                                             'T(MP)', 'T(DSP)',
                                                                                             'T(TOT)', 'T(CUM)',
                                                                                             '# CUTS'))
            print('=============================================='
                  '======================================================')
        else:
            print('{:<4}  {:>15}  {:>15}  {:>9}%  {:>8}s  {:>6}s  {:>6}s  {:>6}s  {:>10}'.format(
                self.iteration, round(self.obj_mp, 3), round(self.obj_dsp, 3), self.gap,
                round(self.run_time_mp, 3), round(self.run_time_dsp, 3),
                round((self.run_time_mp + self.run_time_dsp), 2), round(self.run_time, 2), self.number_cuts))

    def write(self):
        # write solution performance for instance (scenario_code) and solver info (solver_param_code)
        if self.iteration == 0:
            return

        if self.scenario_code is not None and self.solver_param_code is not None:
            cols = ['ITER', 'V(MP)', 'V(DSP)', 'GAP', 'OBJ_VAL',
                    'T(MP)', 'T(DSP)', 'T(TOT)', 'T(CUM)', '# CUTS', 'TIME_LIMIT_REACHED',
                    '# SHORTEST_PATHS', '# NODES', '# EDGES', '# ODS']
            df = pd.DataFrame(data=[[self.gap, self.obj_val, self.run_time, self.number_cuts,
                                     int(self.time_limit_reached), self.num_shortest_paths, self.G.number_of_nodes(),
                                     self.G.number_of_edges(), len(self.od_list)]],
                              columns=['GAP', 'OBJ_VAL', 'T(CUM)', '# CUTS', 'TIME_LIMIT_REACHED',
                                       '# SHORTEST_PATHS', '# NODES', '# EDGES', '# ODS'],
                              index=range(len(self.logged_iterations) + 1))
            for i, log_iter_items in enumerate(self.logged_iterations):
                df.loc[i + 1, cols] = log_iter_items

            df[cols].to_csv(os.path.join(EXP_DIR, self.scenario_code + '_' + self.solver_param_code + '.csv'))

    def solve(self):
        # call the correct solver method depending on input parameters (max_flow, deviation_paths, etc)
        # store the returned values in self.y_val, self.z_val, etc.
        self.pre_process_parameters()

        # TODO: include any additional/alternative solution methods as needed (min cost, p2p, etc.) and divert as needed
        if self.deviation_paths:
            if self.max_flow:
                self.initialize_mp_dc_frlm_max_benders()
                self.solve_mp_dc_frlm_max()

    def solve_mp_dc_frlm_max(self):
        # implementation of mp_dc_frlm_max for benders, return necessary values for storage in G
        # should call methods for master, dual, primal, etc. in iterative process
        # should make use of class variables and update these as iterations proceed

        # while z_mp > z_dsp + eps and g < MAX_ITER:
        while (1 - self.solution_tol) * self.obj_mp > self.obj_dsp and self.iteration < self.max_iter:

            if self.iteration == 0:
                if self.strong_cuts:
                    # if first iteration, solve original dual with interior point value <y_ip_val>
                    # self.solve_dual_primal_sp_analytic(self.y_ip_val)
                    self.solve_dual_primal_sp_analytic_mw()
                    # should not update the dual objective value (keep initialized value)
                    # self.obj_dsp = self.obj_psp
                    self.run_time += self.run_time_dsp
                else:
                    self.solve_dual_primal_sp_analytic(self.y_val)
                    self.run_time += self.run_time_dsp

            else:
                # solve master problem
                self.solve_master()
                # time limit reached and cut off master problem
                if self.time_limit_reached:
                    # end BD algorithm here
                    break

                self.run_time += self.run_time_mp

                if self.strong_cuts:
                    # update interior point values
                    for t in self.time_horizon:
                        self.y_ip_val[t] = (self.y_val[t] + self.y_ip_val[t]) / 2
                    # solve MW dual subproblem (to add Benders' cuts to master problem)
                    self.solve_dual_primal_sp_analytic_mw()
                    self.run_time += self.run_time_dsp
                else:
                    # solve standard dual subproblem
                    self.solve_dual_primal_sp_analytic(self.y_val)
                    self.run_time += self.run_time_dsp

                # calculate and update solution gap
                self.gap = round(100 * (self.obj_mp - self.obj_dsp) / self.obj_mp, 3)

            # print solution status
            self.print_iteration_status()
            # log iteration status
            self.log_iteration()
            # write log to csv
            self.write()

            # update remaining solution time
            if self.solution_time_limit is not None:
                self.solution_time_limit = np.max(
                    [0, self.solution_time_limit - self.run_time_mp - self.run_time_dsp])

            # increase iteration counter
            self.iteration += 1
            # reset runtime storage variables
            self.run_time_dsp = 0
            self.run_time_mp = 0

        if self.time_limit_reached and self.print_status:
                print('\t TIME LIMIT REACHED:: V(MP) within {v0}% of V(DSP) after {v1} iterations'.format(
                    v0=round(100 * (self.obj_mp - self.obj_dsp) / self.obj_mp, 3), v1=self.iteration))

        self.solve_primal_sp_analytic()

        self.run_time += self.run_time_psp
        self.obj_val = self.obj_psp

        # if z_mp <= z_dsp + eps:
        if (1 - self.solution_tol) * self.obj_mp <= self.obj_dsp and self.print_status:
            print('\t THRESHOLD (DUAL) REACHED:: V(MP) within {v0}% of V(DSP) ({v1} ~= {v2})'.format(
                v0=100 * self.solution_tol,
                v1=self.obj_mp, v2=self.obj_dsp))
        elif self.print_status:
            print(f'\t MAX ITERATIONS:: {self.iteration}')

        if self.print_status:
            print('\t THRESHOLD (PRIMAL) CHECK:: V(MP) = {v1} ~<= {v2} = V(PSP)'.format(v1=self.obj_mp,
                                                                                        v2=self.obj_psp))

    def solve_master(self):
        # solve master problem
        # should make use of class variables and update these as iterations proceed
        #  - (think of any clever ways to update matrix values that reduce computation time)
        one_vec_T = np.ones((len(self.od_list),)).T

        if self.model_mp is None:
            # set up model
            self.model_mp = gp.Model('Master Problem', env=gurobi_suppress_output(self.suppress_output))

            # facility location and flow selection variables
            self.y_model_mp = {t: self.model_mp.addMVar((self.n,), vtype=GRB.BINARY, lb=self.y_lb[t], ub=self.y_ub[t],
                                                        name='y_{v1}_MP'.format(v1=t)) for t in self.time_horizon}

            # eta variable (continuous)
            if self.multi_cut_time and self.multi_cut_ods:
                self.eta_model_mp = {t: {
                    od_idx: self.model_mp.addVar(vtype=GRB.CONTINUOUS, name='eta_{v1}_{v2}_MP'.format(v1=od_idx, v2=t))
                    for od_idx, _ in enumerate(self.od_list)} for t in self.time_horizon}
            elif self.multi_cut_time:
                self.eta_model_mp = {t: self.model_mp.addVar(vtype=GRB.CONTINUOUS, name='eta_{v1}_MP'.format(v1=t))
                                     for t in self.time_horizon}
            elif self.multi_cut_ods:
                self.eta_model_mp = {od_idx: self.model_mp.addVar(vtype=GRB.CONTINUOUS,
                                                                  name='eta_{v1}_MP'.format(v1=od_idx))
                                     for od_idx, _ in enumerate(self.od_list)}
            else:
                self.eta_model_mp = self.model_mp.addVar(vtype=GRB.CONTINUOUS, name='eta_agg_MP')

            # objective fxn
            if self.multi_cut_time and self.multi_cut_ods:
                self.model_mp.setObjective(gp.quicksum(
                    self.discount_rates[t] * gp.quicksum(
                        self.eta_model_mp[t][od_idx] for od_idx, _ in enumerate(self.od_list))
                    for t in self.time_horizon), GRB.MAXIMIZE)
            elif self.multi_cut_time:
                self.model_mp.setObjective(
                    gp.quicksum(self.discount_rates[t] * self.eta_model_mp[t] for t in self.time_horizon), GRB.MAXIMIZE)
            elif self.multi_cut_ods:
                self.model_mp.setObjective(
                    gp.quicksum(self.eta_model_mp[od_idx] for od_idx, _ in enumerate(self.od_list)), GRB.MAXIMIZE)
            else:
                self.model_mp.setObjective(self.eta_model_mp, GRB.MAXIMIZE)

            # constraints + benders' cuts
            for g in range(len(self.l_vals)):
                # disaggregate (applied for each time period and demand)
                if self.multi_cut_time and self.multi_cut_ods:
                    for t in self.time_horizon:
                        for od_idx, (o, d) in enumerate(self.od_list):
                            npod_idxs = self.od_npod_idxs[(o, d)]
                            npdo_idxs = self.od_npdo_idxs[(o, d)]
                            C = self.model_mp.addConstr(
                                ((self.A_out[self.range_km[t]].T[:, npod_idxs] @ self.l_vals[g][t][npod_idxs]).T @
                                 self.y_model_mp[t] +
                                 (self.A_rev[self.range_km[t]].T[:, npdo_idxs] @ self.mu_vals[g][t][npdo_idxs]).T
                                 @ self.y_model_mp[t] +
                                 self.r_vals[g][t][od_idx]) >= self.eta_model_mp[t][od_idx],
                                name='MP_C2_{v1}_{v2}_{v3}'.format(v1=od_idx, v2=t, v3=g))
                            C.setAttr('Lazy', 1)
                            # track number of cuts
                            self.number_cuts += 1
                # disaggregate (applied for each time period)
                elif self.multi_cut_time:
                    for t in self.time_horizon:
                        C = self.model_mp.addConstr(
                            (self.A_out[self.range_km[t]].T @ self.l_vals[g][t]).T @ self.y_model_mp[t] +
                            (self.A_rev[self.range_km[t]].T @ self.mu_vals[g][t]).T @ self.y_model_mp[t] +
                            one_vec_T @ self.r_vals[g][t] >= self.eta_model_mp[t],
                            name='MP_C2_{v1}_{v2}'.format(v1=t, v2=g))
                        C.setAttr('Lazy', 1)
                        # track number of cuts
                        self.number_cuts += 1
                # disaggregate (applied for each demand)
                elif self.multi_cut_ods:
                    for od_idx, (o, d) in enumerate(self.od_list):
                        npod_idxs = self.od_npod_idxs[(o, d)]
                        npdo_idxs = self.od_npdo_idxs[(o, d)]
                        C = self.model_mp.addConstr(
                            sum(self.discount_rates[t] *
                                ((self.A_out[self.range_km[t]].T[:, npod_idxs] @
                                  self.l_vals[g][t][npod_idxs]).T @ self.y_model_mp[t] +
                                 (self.A_rev[self.range_km[t]].T[:, npdo_idxs] @
                                  self.mu_vals[g][t][npdo_idxs]).T @ self.y_model_mp[t] +
                                 self.r_vals[g][t][od_idx]) for t in self.time_horizon) >= self.eta_model_mp[od_idx],
                            name='MP_C2_{v1}_{v2}'.format(v1=od_idx, v2=g))
                        C.setAttr('Lazy', 1)
                        # track number of cuts
                        self.number_cuts += 1
                # aggregate cuts (summed over all time periods w/ discount factor)
                else:
                    C = self.model_mp.addConstr(
                        sum(self.discount_rates[t] *
                            ((self.A_out[self.range_km[t]].T @ self.l_vals[g][t]).T @ self.y_model_mp[t] +
                             (self.A_rev[self.range_km[t]].T @ self.mu_vals[g][t]).T @ self.y_model_mp[t] +
                             one_vec_T @ self.r_vals[g][t]) for t in self.time_horizon) >= self.eta_model_mp,
                        name='MP_C2_agg_{v1}'.format(v1=g))
                    C.setAttr('Lazy', 1)
                    # track number of cuts
                    self.number_cuts += 1

            for t_idx, t in enumerate(self.time_horizon):
                # budget constraints
                if t_idx == 0:
                    self.model_mp.addConstr(self.facility_costs[t] @ self.y_model_mp[t] <= self.budgets[t],
                                            name='MP_C3_{v1}'.format(v1=t))
                else:
                    self.model_mp.addConstr(self.facility_costs[t] @ self.y_model_mp[t] -
                                            self.facility_costs[t] @ self.y_model_mp[self.time_horizon[t_idx - 1]] <=
                                            self.budgets[t],
                                            name='MP_C3_{v1}'.format(v1=t))

                # facility nestedness constraints
                if self.nested:
                    if t != self.time_horizon[-1]:
                        self.model_mp.addConstr(self.y_model_mp[t] <= self.y_model_mp[self.time_horizon[t_idx + 1]],
                                                name='MP_C4_{v1}'.format(v1=t))
        # else if model package already instantiated, just add cuts to it
        else:
            # disaggregate (applied for each time period and demand)
            if self.multi_cut_time and self.multi_cut_ods:
                for t in self.time_horizon:
                    for od_idx, (o, d) in enumerate(self.od_list):
                        npod_idxs = self.od_npod_idxs[(o, d)]
                        npdo_idxs = self.od_npdo_idxs[(o, d)]
                        C = self.model_mp.addConstr(
                            ((self.A_out[self.range_km[t]].T[:, npod_idxs] @ self.l_vals[-1][t][npod_idxs]).T @
                             self.y_model_mp[t] +
                             (self.A_rev[self.range_km[t]].T[:, npdo_idxs] @ self.mu_vals[-1][t][npdo_idxs]).T
                             @ self.y_model_mp[t] +
                             self.r_vals[-1][t][od_idx]) >= self.eta_model_mp[t][od_idx],
                            name='MP_C2_{v1}_{v2}_{v3}'.format(v1=od_idx, v2=t, v3=len(self.l_vals) - 1))
                        C.setAttr('Lazy', 1)
                        # track number of cuts
                        self.number_cuts += 1
            # disaggregate (applied for each time period)
            elif self.multi_cut_time:
                for t in self.time_horizon:
                    C = self.model_mp.addConstr(
                        (self.A_out[self.range_km[t]].T @ self.l_vals[-1][t]).T @ self.y_model_mp[t] +
                        (self.A_rev[self.range_km[t]].T @ self.mu_vals[-1][t]).T @ self.y_model_mp[t] +
                        one_vec_T @ self.r_vals[-1][t] >= self.eta_model_mp[t],
                        name='MP_C2_{v1}_{v2}'.format(v1=t, v2=len(self.l_vals) - 1))
                    C.setAttr('Lazy', 1)
                    # track number of cuts
                    self.number_cuts += 1
            # disaggregate (applied for each demand)
            elif self.multi_cut_ods:
                for od_idx, (o, d) in enumerate(self.od_list):
                    npod_idxs = self.od_npod_idxs[(o, d)]
                    npdo_idxs = self.od_npdo_idxs[(o, d)]
                    C = self.model_mp.addConstr(
                        sum(self.discount_rates[t] *
                            ((self.A_out[self.range_km[t]].T[:, npod_idxs] @
                              self.l_vals[-1][t][npod_idxs]).T @ self.y_model_mp[t] +
                             (self.A_rev[self.range_km[t]].T[:, npdo_idxs] @
                              self.mu_vals[-1][t][npdo_idxs]).T @ self.y_model_mp[t] +
                             self.r_vals[-1][t][od_idx]) for t in self.time_horizon) >= self.eta_model_mp[od_idx],
                        name='MP_C2_{v1}_{v2}'.format(v1=od_idx, v2=len(self.l_vals) - 1))
                    C.setAttr('Lazy', 1)
                    # track number of cuts
                    self.number_cuts += 1
            # aggregate cuts (summed over all time periods w/ discount factor)
            else:
                C = self.model_mp.addConstr(
                    sum(self.discount_rates[t] *
                        ((self.A_out[self.range_km[t]].T @ self.l_vals[-1][t]).T @ self.y_model_mp[t] +
                         (self.A_rev[self.range_km[t]].T @ self.mu_vals[-1][t]).T @ self.y_model_mp[t] +
                         one_vec_T @ self.r_vals[-1][t]) for t in self.time_horizon) >= self.eta_model_mp,
                    name='MP_C2_agg_{v1}'.format(v1=len(self.l_vals) - 1))
                C.setAttr('Lazy', 1)
                # track number of cuts
                self.number_cuts += 1

        if self.solution_time_limit is not None:
            self.model_mp.setParam('TimeLimit', self.solution_time_limit)

        t1 = time.time()
        # optimize
        self.model_mp.optimize()
        self.run_time_mp = time.time() - t1

        # if time limit exceeded
        if self.model_mp.Status == 9:
            self.time_limit_reached = True
        else:
            # extract solution values
            self.y_val = {t: self.y_model_mp[t].X for t in self.time_horizon}
            # clean y_val in case there is some strange rounding issues
            for t in self.time_horizon:
                self.y_val[t][self.y_val[t] < self.solution_tol] = 0
                self.y_val[t][self.y_val[t] > 0] = 1

            if self.multi_cut_time and self.multi_cut_ods:
                self.eta_val = {t: {od_idx: self.eta_model_mp[t][od_idx].X for od_idx, _ in enumerate(self.od_list)}
                                for t in self.time_horizon}
            elif self.multi_cut_time:
                self.eta_val = {t: self.eta_model_mp[t].X for t in self.time_horizon}
            elif self.multi_cut_ods:
                self.eta_val = {od_idx: self.eta_model_mp[od_idx].X for od_idx, _ in enumerate(self.od_list)}
            else:
                self.eta_val = self.eta_model_mp.X

            self.obj_mp = self.model_mp.ObjVal  # get objective fxn value

    def solve_dual_primal_sp_analytic(self, y_val):

        # dual variables
        l_val = {t: np.zeros((len(self.npod_comb),)) for t in self.time_horizon}
        mu_val = {t: np.zeros((len(self.npdo_comb),)) for t in self.time_horizon}
        v_val = {t: np.zeros((len(self.od_list),)) for t in self.time_horizon}
        r_val = {t: np.zeros((len(self.od_list),)) for t in self.time_horizon}

        obj_val = 0
        t_solve = 0
        # TODO: can create a subroutine for this that is parametrized on <t>;
        #  can be used for parallelization;
        #  this function would just do the assignment of the results for each time period
        for t in self.time_horizon:
            A_out_y = self.A_out[self.range_km[t]] @ y_val[t]
            A_rev_y = self.A_rev[self.range_km[t]] @ y_val[t]

            A_out_y_norm = A_out_y.copy()
            A_out_y_norm[A_out_y_norm > 0] = 1
            A_out_y_norm = csr_matrix(A_out_y_norm).T
            A_rev_y_norm = A_rev_y.copy()
            A_rev_y_norm[A_rev_y_norm > 0] = 1
            A_rev_y_norm = csr_matrix(A_rev_y_norm).T

            t0 = time.time()

            b_out = self.H_out.T @ (A_out_y_norm - (self.H_out @ np.ones((len(self.pod_comb), 1))))
            b_rev = self.H_rev.T @ (A_rev_y_norm - (self.H_rev @ np.ones((len(self.pdo_comb), 1))))

            # TODO: can even further create a subroutine for this that is parametrized on <o, d>;
            # for each OD pair (index)
            for od_idx, (o, d) in enumerate(self.od_list):
                # if this OD pair is servable by at least 1 path in both the outbound and return directions
                # if servable_ods_out[od_idx] >= 0 and servable_ods_rev[od_idx] >= 0:
                p_outs = self.od_ps_dict[o, d]
                pod_out_idxs = [self.pod_idx_dict[(p, (o, d))] for p in p_outs]
                p_revs = self.do_ps_dict[d, o]
                pdo_rev_idxs = [self.pdo_idx_dict[(p, (d, o))] for p in p_revs]
                # if at least one reverse path is servable for this OD pair
                pdo_idx_max = pdo_rev_idxs[np.argmax(b_rev[pdo_rev_idxs])]
                # if no return path is servable (equivalent to <sum(w_val[t][pdo_rev_idxs]) == 0>)
                if b_rev[pdo_idx_max] < 0:
                    # if sum(w_val[t][pdo_rev_idxs]) == 0:
                    # find the maximum flow value
                    max_flow_val = max(self.pod_flows[t][pod_out_idxs, 0])
                    # assign <max_flow_val> to v_val for this OD pair and time
                    v_val[t][od_idx] = max_flow_val
                    # currently, for each <p> in <(p, od)> in <pod_comb> we pick an
                    #  arbitrary npod_idx with A_rev_y_norm[npod_idx] == 0 and set the corresponding mu[npod_idx] = 0;
                    #  - for MW: we may select it to be npod_idx = argmin{A_rev_y_hat_norm[npod_ix]},
                    #     where A_rev_y_hat_norm is A_rev @ y_hat, and y_hat is an interior point of the current iteration
                    # assign <max_flow_val> to an arbitrary mu[npdo_idx] s.t. npdo_idx = argmin{A_rev_y_norm[npdo_idx]} = 0
                    #  for each <p> in <(p, do)>
                    for p in p_revs:
                        # extract the corresponding <npdo_idxs> s.t. <(p, do)> coincides with <(i, p, r, do)> in <npdo_idxs>
                        npdo_idxs = [self.npdo_idx_dict[npdo] for npdo in self.dop_npdos_dict[d, o][p]]
                        # find the index of the min-valued RHS of the constraint
                        npdo_idx_min = npdo_idxs[np.argmin(A_rev_y[npdo_idxs])]
                        # assign the <max_flow_val> to the index corresponding to the min-valued RHS
                        mu_val[t][npdo_idx_min] = max_flow_val
                # elif no outbound path is servable (equivalent to <sum(z_val[t][pod_out_idxs]) == 0>)
                elif max(b_out[pod_out_idxs]) < 0:
                    # elif sum(z_val[t][pod_out_idxs]) == 0:
                    # for each <p> in <(p, od)>, assign pod_flows[pod_idx] to an arbitrary lambda[npod_idx]
                    #  s.t. npod_idx = argmin{A_out_y_norm[npod_idx]} = 0
                    #  for each <p> in <(p, od)>
                    for p in p_outs:
                        # extract the corresponding <npod_idxs> s.t. <(p, od)> coincides with <(i, p, r, od)> in <npod_idxs>
                        npod_idxs = [self.npod_idx_dict[npod] for npod in self.odp_npods_dict[o, d][p]]
                        # find the index of the min-valued RHS of the constraint
                        npod_idx_min = npod_idxs[np.argmin(A_out_y[npod_idxs])]
                        # assign the corresponding <pod_flows> to the index corresponding to the min-valued RHS
                        pod_out_idx = self.pod_idx_dict[(p, (o, d))]
                        l_val[t][npod_idx_min] = self.pod_flows[t][pod_out_idx, 0]
                # this OD pair is servable in both outbound and return directions
                # (equivalent to <sum(z_val[t][pod_out_idxs]) == 1> and <sum(w_val[t][pdo_rev_idxs]) == 1>)
                elif max(b_out[pod_out_idxs]) >= 0 and b_rev[pdo_idx_max] >= 0:
                    # OUTBOUND: select the maximal-flow-valued servable outbound path
                    # for each outbound path (sorted by flow value) that can be used to serve this OD pair
                    sel_pod_idx = 0
                    for pod_idx, pod_flow in sorted(zip(pod_out_idxs, self.pod_flows[t][pod_out_idxs, 0]),
                                                    key=lambda x: x[1], reverse=True):
                        # if the path is servable
                        if b_out[pod_idx] >= 0:
                            # select this path
                            sel_pod_idx = pod_idx
                            break
                    # sel_pod_idx = pod_out_idxs[np.argmax(z_val[t][pod_out_idxs])]
                    sel_flow_val = self.pod_flows[t][sel_pod_idx, 0]
                    v_val[t][od_idx] = sel_flow_val
                    r_val[t][od_idx] = sel_flow_val
                    # obj_val += discount_rates[t] * sel_flow_val
                    #  for each <p> in <(p, od)>
                    for p in p_outs:
                        # extract the corresponding <npod_idxs> s.t. <(p, od)> coincides with <(i, p, r, od)> in <npod_idxs>
                        npod_idxs = [self.npod_idx_dict[npod] for npod in self.odp_npods_dict[o, d][p]]
                        # find the index of the min-valued RHS of the constraint
                        npod_idx_min = npod_idxs[np.argmin(A_out_y[npod_idxs])]
                        # assign the corresponding value to the index corresponding to the min-valued RHS
                        # for f[pod_idx] (i.e., flow of pod_idx): val = max(0, f[pod_idx] - <sel_flow_val>
                        pod_out_idx = self.pod_idx_dict[(p, (o, d))]
                        l_val[t][npod_idx_min] = max(0, self.pod_flows[t][pod_out_idx, 0] - sel_flow_val)

            obj_val += self.discount_rates[t] * (r_val[t] @ np.ones(len(self.od_list, )) +
                                                 A_out_y.T @ l_val[t] + A_rev_y.T @ mu_val[t])

            t_solve += time.time() - t0

        self.l_val = l_val
        self.mu_val = mu_val
        self.v_val = v_val
        self.r_val = r_val

        self.l_vals.append(l_val)
        self.mu_vals.append(mu_val)
        self.v_vals.append(v_val)
        self.r_vals.append(r_val)

        if obj_val >= self.obj_dsp:
            self.obj_dsp = obj_val
        self.run_time_dsp = t_solve
        self.run_time += t_solve

    def solve_dual_primal_sp_analytic_mw(self):

        # dual variables
        l_val = {t: np.zeros((len(self.npod_comb),)) for t in self.time_horizon}
        mu_val = {t: np.zeros((len(self.npdo_comb),)) for t in self.time_horizon}
        v_val = {t: np.zeros((len(self.od_list),)) for t in self.time_horizon}
        r_val = {t: np.zeros((len(self.od_list),)) for t in self.time_horizon}
        # s_val = {t: np.zeros((len(od_list),)) for t in time_horizon}

        obj_val = 0
        t_solve = 0

        for t in self.time_horizon:
            A_out_y = self.A_out[self.range_km[t]] @ self.y_val[t]
            A_out_y_ip = self.A_out[self.range_km[t]] @ self.y_ip_val[t]
            A_rev_y = self.A_rev[self.range_km[t]] @ self.y_val[t]
            A_rev_y_ip = self.A_rev[self.range_km[t]] @ self.y_ip_val[t]

            A_out_y_norm = A_out_y.copy()
            A_out_y_norm[A_out_y_norm > 0] = 1
            A_out_y_norm = csr_matrix(A_out_y_norm).T
            A_rev_y_norm = A_rev_y.copy()
            A_rev_y_norm[A_rev_y_norm > 0] = 1
            A_rev_y_norm = csr_matrix(A_rev_y_norm).T

            t0 = time.time()

            # for checking/creating an optimal primal solution
            b_out = self.H_out.T @ (A_out_y_norm - (self.H_out @ np.ones((len(self.pod_comb), 1))))
            b_rev = self.H_rev.T @ (A_rev_y_norm - (self.H_rev @ np.ones((len(self.pdo_comb), 1))))

            # for each OD pair (index)
            for od_idx, (o, d) in enumerate(self.od_list):
                # if this OD pair is servable by at least 1 path in both the outbound and return directions
                # if servable_ods_out[od_idx] >= 0 and servable_ods_rev[od_idx] >= 0:

                p_outs = self.od_ps_dict[o, d]
                pod_out_idxs = [self.pod_idx_dict[(p, (o, d))] for p in p_outs]
                p_revs = self.do_ps_dict[d, o]
                pdo_rev_idxs = [self.pdo_idx_dict[(p, (d, o))] for p in p_revs]
                # if at least one reverse path is servable for this OD pair
                pdo_idx_max = pdo_rev_idxs[np.argmax(b_rev[pdo_rev_idxs])]
                # if no return path is servable (equivalent to <sum(w_val[t][pdo_rev_idxs]) == 0>)
                if b_rev[pdo_idx_max] < 0:
                    # if no return path is servable
                    # if sum(w_val[t][pdo_rev_idxs]) == 0:
                    # find the maximum flow value
                    max_flow_val = max(self.pod_flows[t][pod_out_idxs, 0])
                    # assign <max_flow_val> to v_val for this OD pair and time
                    v_val[t][od_idx] = max_flow_val
                    # currently, for each <p> in <(p, od)> in <pod_comb> we pick an
                    #  arbitrary npod_idx with A_rev_y_norm[npod_idx] == 0 and set the corresponding mu[npod_idx] = 0;
                    #  - for MW: we may select it to be npod_idx = argmin{A_rev_y_hat_norm[npod_ix]},
                    #     where A_rev_y_hat_norm is A_rev @ y_hat, and y_hat is an interior point of the current iteration
                    # assign <max_flow_val> to an arbitrary mu[npdo_idx] s.t. npdo_idx = argmin{A_rev_y_norm[npdo_idx]} = 0
                    #  for each <p> in <(p, do)>
                    for p in p_revs:
                        # extract the corresponding <npdo_idxs> s.t. <(p, do)> coincides with <(i, p, r, do)> in <npdo_idxs>
                        npdo_idxs = np.array([self.npdo_idx_dict[npdo] for npdo in self.dop_npdos_dict[d, o][p]])
                        # find the indices of all the zero-valued RHS of the constraint evaluated at y_val
                        npdo_idxs_zero = npdo_idxs[np.where(A_rev_y[npdo_idxs] == 0)[0]]
                        # print(A_rev_y[npdo_idxs_zero])
                        # find the index of the min-valued RHS of constraint evaluated at y_ip for the zero-valued indices
                        npdo_idx_min = npdo_idxs_zero[np.argmin(A_rev_y_ip[npdo_idxs_zero])]
                        # print(A_rev_y[npdo_idx_min])
                        # assign the <max_flow_val> to the index corresponding to the min-valued RHS
                        mu_val[t][npdo_idx_min] = max_flow_val
                # elif no outbound path is servable (equivalent to <sum(z_val[t][pod_out_idxs]) == 0>)
                elif max(b_out[pod_out_idxs]) < 0:
                    # elif sum(z_val[t][pod_out_idxs]) == 0:
                    # for each <p> in <(p, od)>, assign pod_flows[pod_idx] to an arbitrary lambda[npod_idx]
                    #  s.t. npod_idx = argmin{A_out_y_norm[npod_idx]} = 0
                    #  for each <p> in <(p, od)>
                    # print(sum(w_val[t][pdo_rev_idxs]), sum(z_val[t][pod_out_idxs]))
                    # print(z_val[t][pod_out_idxs])
                    for p in p_outs:
                        # extract the corresponding <npod_idxs> s.t. <(p, od)> coincides with <(i, p, r, od)> in <npod_idxs>
                        npod_idxs = np.array([self.npod_idx_dict[npod] for npod in self.odp_npods_dict[o, d][p]])
                        # find the indices of all the zero-valued RHS of the constraint evaluated at y_val
                        # print(f'z: {z_val[t][pod_idx_dict[(p, (o, d))]]}')
                        # print(f'A: {t, (o, d), p} {A_out_y[npod_idxs]}')
                        npod_idxs_zero = npod_idxs[np.where(A_out_y[npod_idxs] == 0)[0]]
                        # find the index of the min-valued RHS of constraint evaluated at y_ip for the zero-valued indices
                        npod_idx_min = npod_idxs_zero[np.argmin(A_out_y_ip[npod_idxs_zero])]
                        # assign the corresponding <pod_flows> to the index corresponding to the min-valued RHS
                        pod_out_idx = self.pod_idx_dict[(p, (o, d))]
                        l_val[t][npod_idx_min] = self.pod_flows[t][pod_out_idx, 0]
                # this OD pair is servable in both outbound and return directions
                # (equivalent to <sum(z_val[t][pod_out_idxs]) == 1> and <sum(w_val[t][pdo_rev_idxs]) == 1>)
                elif max(b_out[pod_out_idxs]) >= 0 and b_rev[pdo_idx_max] >= 0:
                    # OUTBOUND: select the maximal-flow-valued servable outbound path
                    # for each outbound path (sorted by flow value) that can be used to serve this OD pair
                    sel_pod_idx = 0
                    for pod_idx, pod_flow in sorted(zip(pod_out_idxs, self.pod_flows[t][pod_out_idxs, 0]),
                                                    key=lambda x: x[1], reverse=True):
                        # if the path is servable
                        if b_out[pod_idx] >= 0:
                            # select this path
                            sel_pod_idx = pod_idx
                            break

                    # sel_pod_idx = pod_out_idxs[np.argmax(z_val[t][pod_out_idxs])]
                    sel_flow_val = self.pod_flows[t][sel_pod_idx, 0]
                    v_val[t][od_idx] = sel_flow_val
                    r_val[t][od_idx] = sel_flow_val
                    # obj_val += discount_rates[t] * sel_flow_val
                    #  for each <p> in <(p, od)>
                    for p in p_outs:
                        # extract the corresponding <npod_idxs> s.t. <(p, od)> coincides with <(i, p, r, od)> in <npod_idxs>
                        npod_idxs = np.array([self.npod_idx_dict[npod] for npod in self.odp_npods_dict[o, d][p]])
                        # find the indices of all the zero-valued RHS of the constraint evaluated at y_val
                        npod_idxs_zero = npod_idxs[np.where(A_out_y[npod_idxs] == 0)[0]]
                        # if there are no zero-valued entries on the RHS for this path <p>,
                        # then the lambda value should be kept 0
                        if len(npod_idxs_zero) == 0:
                            continue
                        # find the index of the min-valued RHS of constraint evaluated at y_ip for the zero-valued indices
                        npod_idx_min = npod_idxs_zero[np.argmin(A_out_y_ip[npod_idxs_zero])]
                        # npod_idx_min = npod_idxs[np.argmin(A_out_y[npod_idxs])]
                        # assign the corresponding value to the index corresponding to the min-valued RHS
                        # for f[pod_idx] (i.e., flow of pod_idx): val = max(0, f[pod_idx] - <sel_flow_val>
                        pod_out_idx = self.pod_idx_dict[(p, (o, d))]
                        l_val[t][npod_idx_min] = max(0, self.pod_flows[t][pod_out_idx, 0] - sel_flow_val)

            # objective value for original dual subproblem, evaluated at <y_val> (not the interior point <y_ip_val>)
            obj_val += self.discount_rates[t] * (r_val[t] @ np.ones(len(self.od_list, )) +
                                                 A_out_y.T @ l_val[t] + A_rev_y.T @ mu_val[t])

            t_solve += time.time() - t0

        self.l_val = l_val
        self.mu_val = mu_val
        self.v_val = v_val
        self.r_val = r_val

        self.l_vals.append(l_val)
        self.mu_vals.append(mu_val)
        self.v_vals.append(v_val)
        self.r_vals.append(r_val)

        if obj_val >= self.obj_dsp:
            self.obj_dsp = obj_val
        self.run_time_dsp = t_solve
        self.run_time += t_solve

    def solve_primal_sp_analytic(self):

        # flow selection variables
        z = {t: np.zeros((len(self.pod_comb),)) for t in self.time_horizon}
        w = {t: np.zeros((len(self.pdo_comb),)) for t in self.time_horizon}

        obj_val = 0
        t_solve = 0

        for t in self.time_horizon:
            # needs to be normalized to binary values for below calculation with <b_out> and <b_rev> to work
            A_out_y_norm = self.A_out[self.range_km[t]] @ self.y_val[t]
            A_out_y_norm[A_out_y_norm > 0] = 1
            A_out_y_norm = csr_matrix(A_out_y_norm).T
            A_rev_y_norm = self.A_rev[self.range_km[t]] @ self.y_val[t]
            A_rev_y_norm[A_rev_y_norm > 0] = 1
            A_rev_y_norm = csr_matrix(A_rev_y_norm).T

            t0 = time.time()

            b_out = self.H_out.T @ (A_out_y_norm - (self.H_out @ np.ones((len(self.pod_comb), 1))))
            b_rev = self.H_rev.T @ (A_rev_y_norm - (self.H_rev @ np.ones((len(self.pdo_comb), 1))))

            # for each OD pair (index)
            for od_idx, (o, d) in enumerate(self.od_list):
                # pod_idxs = od_idx_pod_idxs_dict[od_idx]
                # pdo_idxs = od_idx_pdo_idxs_dict[od_idx]
                p_outs = self.od_ps_dict[o, d]
                pod_out_idxs = [self.pod_idx_dict[(p, (o, d))] for p in p_outs]
                p_revs = self.do_ps_dict[d, o]
                pdo_rev_idxs = [self.pdo_idx_dict[(p, (d, o))] for p in p_revs]
                # if at least one reverse path is servable for this OD pair
                pdo_idx_max = pdo_rev_idxs[np.argmax(b_rev[pdo_rev_idxs])]

                if b_rev[pdo_idx_max] >= 0:
                    # RETURN: select any single servable return path
                    # select a single servable return path; the rest will be zero (move on to the next OD pair)
                    w[t][pdo_idx_max] = 1
                    # if this OD pair is servable by at least 1 path the outbound direction as well
                    # this means: servable_ods_out[od_idx] >= 0 and servable_ods_rev[od_idx] >= 0
                    # TODO: NEW
                    # t0 = time.time()
                    # pod_idx_max = pod_out_idxs[np.argmax(b_out[pod_out_idxs])]
                    # # pod_out_idxs_nonneg = np.where(b_out >= 0)[0]
                    # if b_out[pod_idx_max] >= 0:
                    #     z[t][pod_idx_max] = 1
                    #     obj_val += discount_rates[t] * pod_flows[t][pod_idx_max, 0]
                    # print(time.time() - t0)
                    # TODO: END NEW
                    # TODO: UNCOMMENT
                    if max(b_out[pod_out_idxs]) >= 0:
                        # OUTBOUND: select the maximal-flow-valued servable outbound path
                        # for each outbound path (sorted by flow value) that can be used to serve this OD pair
                        for pod_idx, pod_flow in sorted(zip(pod_out_idxs, self.pod_flows[t][pod_out_idxs, 0]),
                                                        key=lambda x: x[1], reverse=True):
                            # if the path is servable
                            if b_out[pod_idx] >= 0:
                                # select this path
                                z[t][pod_idx] = 1
                                # increment the objective value
                                obj_val += self.discount_rates[t] * pod_flow
                                # the rest will be zero (move on to next OD pair)
                                break

            t_solve += time.time() - t0

        self.z_val = z
        self.w_val = w
        self.obj_psp = obj_val
        self.run_time_psp = t_solve
