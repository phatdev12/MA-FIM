import networkx as nx
import numpy as np
import pickle
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator
import math
import community as community_louvain
import sys
import copy
import random
import time
from time import strftime, localtime
import decimal
from decimal import Decimal
import os

def multi_to_set(f, n = None, g_nodes = None):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    if n == None:
        if g_nodes is not None:
            n = len(g_nodes)
        else:
            raise ValueError("Either n or g_nodes must be provided")
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f, i):
    def f_single(x):
        return f(x, 1000)[i]
    return f_single

def pop_init(pop, budget, comm, values, comm_label,nodes_attr,prank):
    P = []

    for _ in range(pop):
        P_it1 = []

        comm_score = {}
        u = {}
        selected_attr = {}

        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        for t in range(len(comm)):
            sco1 = len(comm[t])
            sco2 = 0

            for ca in comm_label[t]:
                sco2 += u[ca]

            comm_score[t] = sco1 * sco2

        comm_sel = {}

        for _ in range(budget):
            a = list(comm_score.keys())#comm number
            b = list(comm_score.values())#score

            b_sum = sum(b)
            for deg in range(len(b)):
                b[deg] /= b_sum
            b = np.array(b)
            tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

            if tar_comm in list(comm_sel.keys()):
                comm_sel[tar_comm] += 1
            else:
                comm_sel[tar_comm] = 1
                for att in comm_label[tar_comm]:
                    selected_attr[att] += len(set(nodes_attr[att])&set(comm[tar_comm]))
                    u[att] = math.exp(-1*selected_attr[att]/len(nodes_attr[att]))

            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2

        for cn in list(comm_sel.keys()):
            pr = {}
            for nod in comm[cn]:
                pr[nod] = prank[nod]

            pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            for pr_ind in range(comm_sel[cn]):
                P_it1.append(pr[pr_ind][0])

        P.append(P_it1)

    return P

def crossover(P1, cr, budget, partition, comm_label, comm, values, nodes_attr, prank):
    P = copy.deepcopy(P1)

    for i in range(int(len(P)/2)):
        for j in range(len(P[i])):
            if random.random() < cr:
                temp = P[i][j]
                P[i][j] = P[len(P)-i-1][j]
                P[len(P)-i-1][j] = temp

    for i in range(len(P)):
        P[i] = list(set(P[i]))
        if len(P[i]) == budget:
            continue

        comm_score = {}
        u = {}
        selected_attr = {}
        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        all_comm = []
        for node in P[i]:
            all_comm.append(partition[node])
        all_comm = list(set(all_comm))

        for ac in all_comm:
            for ca in comm_label[ac]:
                selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

        for t in range(len(comm)):
            sco1 = len(comm[t])
            sco2 = 0

            for ca in comm_label[t]:
                sco2 += u[ca]

            comm_score[t] = sco1 * sco2

        while len(P[i])<budget:
            a = list(comm_score.keys())  # comm number
            b = list(comm_score.values())  # score

            b_sum = sum(b)
            for deg in range(len(b)):
                b[deg] /= b_sum
            b = np.array(b)
            tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

            if tar_comm not in all_comm:
                all_comm.append(tar_comm)

                for ca in comm_label[tar_comm]:
                    selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[tar_comm]))
                    u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

            pr = {}
            for nod in comm[tar_comm]:
                pr[nod] = prank[nod]

            aa = list(pr.keys())
            bb = list(pr.values())

            bb_sum = sum(bb)
            for deg in range(len(bb)):
                bb[deg] /= bb_sum
            bb = np.array(bb)

            while True:
                tar_node = np.random.choice(aa, size=1, p=bb.ravel())[0]
                if tar_node not in P[i]:
                    P[i].append(tar_node)
                    break

            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2

    return P

def mutation(P1, mu, comm, values, nodes_attr, prank, partition, comm_label):
    P = copy.deepcopy(P1)

    for i in range(len(P)):
        for j in range(len(P[i])):
            if random.random() < mu:
                comm_score = {}
                u = {}
                selected_attr = {}
                for cal in values:
                    u[cal] = 1
                    selected_attr[cal] = 0

                all_comm = []
                for node in P[i]:
                    all_comm.append(partition[node])
                all_comm.remove(partition[P[i][j]])
                all_comm = list(set(all_comm))

                for ac in all_comm:
                    for ca in comm_label[ac]:
                        selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                        u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

                for t in range(len(comm)):
                    sco1 = len(comm[t])
                    sco2 = 0

                    for ca in comm_label[t]:
                        sco2 += u[ca]

                    comm_score[t] = sco1 * sco2

                a = list(comm_score.keys())  # comm number
                b = list(comm_score.values())  # score

                b_sum = sum(b)
                for deg in range(len(b)):
                    b[deg] /= b_sum
                b = np.array(b)
                tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]


                pr = {}
                for nod in comm[tar_comm]:
                    pr[nod] = prank[nod]

                aa = list(pr.keys())
                bb = list(pr.values())

                bb_sum = sum(bb)
                for deg in range(len(bb)):
                    bb[deg] /= bb_sum
                bb = np.array(bb)

                while True:
                    tar_node = np.random.choice(aa, size=1, p=bb.ravel())[0]
                    if tar_node not in P[i]:
                        P[i][j] = tar_node
                        break

    return P

def local_search_in_loop(solution, Eval, partition, prank, max_iterations=10):
    best_solution = copy.deepcopy(solution)
    best_fitness = Eval(best_solution)
    
    all_nodes = list(partition.keys())
    
    for iteration in range(max_iterations):
        improved = False
        
        # Chỉ lấy top 5 candidates (nhanh hơn)
        candidates = [n for n in all_nodes if n not in best_solution]
        candidates.sort(key=lambda x: prank[x], reverse=True)
        top_candidates = candidates[:5]
        
        # Thử thay thế từng vị trí
        for i in range(len(best_solution)):
            for candidate in top_candidates:
                test_solution = copy.deepcopy(best_solution)
                test_solution[i] = candidate
                test_fitness = Eval(test_solution)
                
                if test_fitness > best_fitness:
                    best_fitness = test_fitness
                    best_solution = test_solution
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_solution

def final_local_search_simple(solution, Eval, partition, prank, max_iterations=30):
    best_solution = copy.deepcopy(solution)
    best_fitness = Eval(best_solution)
    
    all_nodes = list(partition.keys())
    
    print(f"Final Local Search - Initial: {best_fitness:.6f}")
    
    for iteration in range(max_iterations):
        improved = False
        
        # Lấy top 10 candidates (nhiều hơn cái trong loop)
        candidates = [n for n in all_nodes if n not in best_solution]
        candidates.sort(key=lambda x: prank[x], reverse=True)
        top_candidates = candidates[:10]
        
        # Thử thay thế từng vị trí
        for i in range(len(best_solution)):
            for candidate in top_candidates:
                test_solution = copy.deepcopy(best_solution)
                test_solution[i] = candidate
                test_fitness = Eval(test_solution)
                
                if test_fitness > best_fitness:
                    best_fitness = test_fitness
                    best_solution = test_solution
                    improved = True
                    print(f"  Iter {iteration+1}: Improved -> {best_fitness:.6f}")
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    print(f"Final: {best_fitness:.6f}")
    return best_solution


succession = True
solver = 'md'

group_size = {}
num_runs = 20
algorithms = ['Greedy', 'GR', 'MaxMin-Size']

# graphnames = ['graph_spa_500_0']
# attributes = ['region', 'ethnicity', 'age', 'gender', 'status']

def run(graphnames, attributes):
  # Create output directory if it doesn't exist
  output_dir = 'results'
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # Create output file with timestamp and dataset name
  timestamp = strftime("%Y%m%d_%H%M%S", localtime())
  dataset_name = f"{graphnames[0]}_{attributes[0]}"  # e.g., "rice_subset_color"
  output_file = os.path.join(output_dir, f'Better_CEA_FIM_{dataset_name}_{timestamp}.txt')

  # Open output file for writing
  with open(output_file, 'w') as f_out:
      f_out.write("="*80 + "\n")
      f_out.write("Better CEA-FIM Algorithm Results\n")
      f_out.write(f"Dataset: {dataset_name}\n")
      f_out.write(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S', localtime())}\n")
      f_out.write("="*80 + "\n\n")

  for graphname in graphnames:
      print(graphname)
      # Also write to file
      with open(output_file, 'a') as f_out:
          f_out.write(f"\nGraph: {graphname}\n")
          f_out.write("-"*80 + "\n")
      for budget in [40]:
          g = pickle.load(open('networks/{}.pickle'.format(graphname), 'rb'))
          ng = list(g.nodes())
          ngIndex = {}
          for ni in range(len(ng)):
              ngIndex[ng[ni]] = ni

          # propagation probability for the ICM
          p = 0.01
          for u, v in g.edges():
              g[u][v]['p'] = p

          g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

          group_size[graphname] = {}

          for attribute in attributes:
              # assign a unique numeric value for nodes who left the attribute blank
              nvalues = len(np.unique([g.nodes[v][attribute] for v in g.nodes()]))
              group_size[graphname][attribute] = np.zeros((num_runs, nvalues))

          fair_vals_attr = np.zeros((num_runs, len(attributes)))
          greedy_vals_attr = np.zeros((num_runs, len(attributes)))
          pof = np.zeros((num_runs, len(attributes)))

          include_total = False

          for attr_idx, attribute in enumerate(attributes):

              live_graphs = sample_live_icm(g, 1000)

              group_indicator = np.ones((len(g.nodes()), 1))

              val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                    list(g.nodes()), np.ones(len(g)))

              def f_multi(x):
                  return val_oracle(x, 1000).sum()

              f_set = multi_to_set(f_multi, g_nodes=g.nodes())

              violation_0 = []
              violation_1 = []
              min_fraction_0 = []
              min_fraction_1 = []
              pof_0 = []
              time_0 = []
              time_1 = []

              alpha = 0.5  # a*MF+(1-a)*DCV
              print('alpha ', alpha)
              with open(output_file, 'a') as f_out:
                  f_out.write(f"Alpha: {alpha}\n")
                  f_out.write(f"Attribute: {attribute}\n")
                  f_out.write(f"Budget: {budget}\n\n")

              for run in range(num_runs):
                  print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
                  with open(output_file, 'a') as f_out:
                      f_out.write(f"\nRun {run+1}/{num_runs} - {strftime('%Y-%m-%d %H:%M:%S', localtime())}\n")
                  # find overall optimal solution
                  start_time1 = time.perf_counter()
                  S, obj = greedy(list(range(len(g))), budget, f_set)
                  end_time1 = time.perf_counter()
                  runningtime1 = end_time1 - start_time1

                  start_time = time.perf_counter()
                  # all values taken by this attribute
                  values = np.unique([g.nodes[v][attribute] for v in g.nodes()])

                  nodes_attr = {}  # value-node

                  for vidx, val in enumerate(values):
                      nodes_attr[val] = [v for v in g.nodes() if g.nodes[v][attribute] == val]
                      group_size[graphname][attribute][run, vidx] = len(nodes_attr[val])

                  opt_succession = {}
                  if succession:
                      for vidx, val in enumerate(values):
                          h = nx.subgraph(g, nodes_attr[val])
                          h = nx.convert_node_labels_to_integers(h)
                          live_graphs_h = sample_live_icm(h, 1000)
                          group_indicator = np.ones((len(h.nodes()), 1))
                          val_oracle = multi_to_set(valoracle_to_single(
                              make_multilinear_objective_samples_group(live_graphs_h, group_indicator, list(h.nodes()),
                                                                      list(h.nodes()), np.ones(len(h))), 0), len(h))
                          S_succession, opt_succession[val] = greedy(list(h.nodes()),
                                                                    math.ceil(len(nodes_attr[val]) / len(g) * budget),
                                                                    val_oracle)

                  if include_total:
                      group_indicator = np.zeros((len(g.nodes()), len(values) + 1))
                      for val_idx, val in enumerate(values):
                          group_indicator[nodes_attr[val], val_idx] = 1
                      group_indicator[:, -1] = 1
                  else:
                      group_indicator = np.zeros((len(g.nodes()), len(values)))
                      for val_idx, val in enumerate(values):
                          group_indicator[nodes_attr[val], val_idx] = 1

                  val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                        list(g.nodes()), np.ones(len(g)))

                  # build an objective function for each subgroup
                  f_attr = {}
                  f_multi_attr = {}
                  for vidx, val in enumerate(values):
                      nodes_attr[val] = [v for v in g.nodes() if g.nodes[v][attribute] == val]
                      f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
                      f_attr[val] = multi_to_set(f_multi_attr[val], g_nodes=g.nodes())

                  # get the best seed set for nodes of each subgroup
                  S_attr = {}
                  opt_attr = {}
                  if not succession:
                      for val in values:
                          S_attr[val], opt_attr[val] = greedy(list(range(len(g))),
                                                              int(len(nodes_attr[val]) / len(g) * budget), f_attr[val])
                  if succession:
                      opt_attr = opt_succession
                  all_opt = np.array([opt_attr[val] for val in values])


                  def Eval(SS):
                      S = [ngIndex[int(i)] for i in SS]
                      fitness = 0
                      x = np.zeros(len(g.nodes))
                      x[list(S)] = 1

                      vals = val_oracle(x, 1000)
                      coverage_min = (vals / group_size[graphname][attribute][run]).min()
                      violation = np.clip(all_opt - vals, 0, np.inf) / all_opt

                      fitness += alpha * coverage_min
                      fitness -= (1-alpha) * violation.sum() / len(values)

                      return fitness


                  # EA-start
                  pop = 10
                  mu = 0.1
                  cr = 0.6
                  maxgen = 150

                  address = 'networks/{}.txt'.format(graphname)
                  G = nx.read_edgelist(address, create_using=nx.Graph())

                  partition = community_louvain.best_partition(G)
                  comm_all_label = list(set(partition.values()))#社团标签，非节点
                  comm = []
                  for _ in range(len(comm_all_label)):
                      comm.append([])
                  for key in list(partition.keys()):
                      comm[partition[key]].append(key)

                  comm_label = []#每个社团含有的节点属性
                  for c in comm:
                      temp = set()
                      for cc in c:
                          temp.add(g.nodes[ngIndex[int(cc)]][attribute])
                      comm_label.append(list(temp))

                  pr = nx.pagerank(G)

                  P = pop_init(pop, budget, comm, values,comm_label,nodes_attr,pr)

                  i = 0
                  while i < maxgen:
                      P = sorted(P, key=lambda x: Eval(x), reverse=True)

                      P_cr = crossover(P, cr, budget, partition, comm_label, comm, values, nodes_attr, pr)
                      P_mu = mutation(P, mu, comm, values, nodes_attr, pr, partition, comm_label)

                      for index in range(pop):
                          inf1 = Eval(P_mu[index])
                          inf2 = Eval(P[index])

                          if inf1 > inf2:
                              P[index] = P_mu[index]
                      
                      best_idx = 0
                      P[best_idx] = local_search_in_loop(P[best_idx], Eval, partition, pr, max_iterations=10)
                      
                      i += 1

                  SS = sorted(P, key=lambda x: Eval(x), reverse=True)[0]

                  # ★★★ LOCAL SEARCH 2: SAU KHI EA KẾT THÚC ★★★
                  print("\n" + "="*60)
                  print("Applying Final Local Search (Outside EA Loop)...")
                  print("="*60)
                  SS = final_local_search_simple(
                      solution=SS, 
                      Eval=Eval, 
                      partition=partition, 
                      prank=pr,
                      max_iterations=30
                  )
                  print("="*60)
                  print("Final Local Search completed!\n")

                  SI = [ngIndex[int(si)] for si in SS]

                  # EA-end

                  end_time = time.perf_counter()
                  runningtime = end_time - start_time

                  xg = np.zeros(len(g.nodes))
                  xg[list(S)] = 1

                  fair_x = np.zeros(len(g.nodes))
                  fair_x[list(SI)] = 1

                  greedy_vals = val_oracle(xg, 1000)
                  all_fair_vals = val_oracle(fair_x, 1000)

                  if include_total:
                      greedy_vals = greedy_vals[:-1]
                      all_fair_vals = all_fair_vals[:-1]

                  fair_violation = np.clip(all_opt - all_fair_vals, 0, np.inf) / all_opt
                  greedy_violation = np.clip(all_opt - greedy_vals, 0, np.inf) / all_opt
                  fair_vals_attr[run, attr_idx] = fair_violation.sum() / len(values)
                  greedy_vals_attr[run, attr_idx] = greedy_violation.sum() / len(values)

                  greedy_min = (greedy_vals / group_size[graphname][attribute][run]).min()
                  fair_min = (all_fair_vals / group_size[graphname][attribute][run]).min()

                  pof[run, attr_idx] = greedy_vals.sum() / all_fair_vals.sum()

                  violation_0.append(fair_violation.sum() / len(values))
                  violation_1.append(greedy_violation.sum() / len(values))
                  min_fraction_0.append(fair_min)
                  min_fraction_1.append(greedy_min)
                  pof_0.append(greedy_vals.sum() / all_fair_vals.sum())
                  time_0.append(runningtime)
                  time_1.append(runningtime1)

                  dcv_val = Decimal(fair_violation.sum() / len(values)).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)
                  mf_val = Decimal(fair_min).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)
                  f_val = Decimal(fair_min - fair_violation.sum() / len(values)).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)

                  print("DCV: ", dcv_val)
                  print("MF: ", mf_val)
                  print("F: ", f_val)

                  with open(output_file, 'a') as f_out:
                      f_out.write(f"  DCV: {dcv_val}\n")
                      f_out.write(f"  MF: {mf_val}\n")
                      f_out.write(f"  F: {f_val}\n")
                      f_out.write(f"  Time EA: {runningtime:.2f}s, Time Greedy: {runningtime1:.2f}s\n")

              # Calculate final statistics
              avg_dcv = Decimal(np.mean(violation_0)).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)
              avg_mf = Decimal(np.mean(min_fraction_0)).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)
              avg_f = Decimal(np.mean(min_fraction_0) - np.mean(violation_0)).quantize(Decimal("0.0000"), rounding=decimal.ROUND_HALF_UP)
              avg_viol_ea = Decimal(np.mean(violation_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_viol_greedy = Decimal(np.mean(violation_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_minfra_ea = Decimal(np.mean(min_fraction_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_minfra_greedy = Decimal(np.mean(min_fraction_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_pof = Decimal(np.mean(pof_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_time_ea = Decimal(np.mean(time_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)
              avg_time_greedy = Decimal(np.mean(time_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP)

              print("graph:", graphname, "K:", budget, "attribute", attribute)
              print("DCV: ", avg_dcv)
              print("MF: ", avg_mf)
              print("F:", avg_f)
              print("violation_EA:", avg_viol_ea, "violation_greedy:", avg_viol_greedy)
              print("min_fra_EA:", avg_minfra_ea, "min_fra_greedy:", avg_minfra_greedy)
              print("POF_EA:", avg_pof)
              print("time_EA:", avg_time_ea, "time_greedy:", avg_time_greedy)
              print()

              # Write summary to file
              with open(output_file, 'a') as f_out:
                  f_out.write("\n" + "="*80 + "\n")
                  f_out.write(f"SUMMARY - Graph: {graphname}, K: {budget}, Attribute: {attribute}\n")
                  f_out.write("="*80 + "\n")
                  f_out.write(f"Average DCV: {avg_dcv}\n")
                  f_out.write(f"Average MF: {avg_mf}\n")
                  f_out.write(f"Average F: {avg_f}\n")
                  f_out.write(f"Violation EA: {avg_viol_ea}, Violation Greedy: {avg_viol_greedy}\n")
                  f_out.write(f"Min Fraction EA: {avg_minfra_ea}, Min Fraction Greedy: {avg_minfra_greedy}\n")
                  f_out.write(f"POF EA: {avg_pof}\n")
                  f_out.write(f"Time EA: {avg_time_ea}s, Time Greedy: {avg_time_greedy}s\n")
                  f_out.write("="*80 + "\n\n")

  print(f"\n{'='*80}")
  print(f"Results saved to: {output_file}")
  print(f"{'='*80}")
  
if __name__ == "__main__":
  dataset_graph = [
    (["synth3"], ["color"]),
  ]

  from multiprocessing import Pool
  num_processes = min(6, len(dataset_graph))  # 6 datasets = 6 processes
  print(f"Running {num_processes} datasets in parallel on {num_processes} CPU cores...")
  with Pool(processes=num_processes) as pool:
      pool.starmap(run, dataset_graph)
      
    