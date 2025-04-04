# src/utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import logging
import json

import src.models as models

# ================= IO utils =================================================

def setup_logger(log_file):
    """
    设置 logger，将日志输出到文件和控制台
    """
    logger = logging.getLogger("sweeper_logger")
    logger.setLevel(logging.INFO)

    # 如果已有 handler 则清空，避免重复写入
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器，写入指定文件
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器（可选）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def log_experiment_result(exp_result, logger):
    """
    记录单个实验结果到日志，exp_result 包含：
      - "sweep_param": 当前 sweep 参数值（如 f_max、num_users 等）
      - "node_info": { "X":..., "Y":..., "U_current":..., ... }
      - "user_info": [ { user1 的详细信息 }, { user2 的详细信息 }, ... ]
    """
    logger.info("----- Single Experiment Result -----")
    logger.info("Sweep Parameter: %s", exp_result.get("sweep_param"))
    logger.info("SP Info:\n%s", json.dumps(exp_result.get("sp_info"), ensure_ascii=False, indent=2))
    logger.info("User Game Info:\n%s", json.dumps(exp_result.get("user_game_info"), ensure_ascii=False, indent=2))
    logger.info("User Info:")
    for user in exp_result.get("user_info", []):
        logger.info(json.dumps(user, ensure_ascii=False, indent=2))
    logger.info("----- End of Single Experiment -----\n")

def log_sweep_experiments_results(sweep_results, log_file):
    """
    记录多组实验结果到 log_file，
    sweep_results 是一个列表，每个元素都是一次实验的结果字典
    """
    logger = setup_logger(log_file)
    logger.info("===== Sweep Experiments Results =====")
    for idx, exp_result in enumerate(sweep_results):
        logger.info("===== Experiment %d =====", idx + 1)
        log_experiment_result(exp_result, logger)
    logger.info("===== End of Sweep Experiments Results =====")

def save_results(data, filename):
    # 获取目标目录
    directory = os.path.dirname(filename)
    
    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建")
      
    """
    保存实验结果，data 可以是 dict 或 pandas DataFrame
    """
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    # df.to_csv(filename, index=False)
    df.to_csv(filename, index=False, mode="w", header=True)
    print(f"结果已保存到 {filename}")

def log_message(msg):
    print(f"[LOG] {msg}")

def print_resource_utilization(users, provider, X, p_E, p_N):
  of, ob, oc = models.ora_solver([u for u in users if u.user_id in X], provider, p_E, p_N)
  print(f"f={of},b={ob},c={oc}")
  print(f"sum(f)={np.sum(of)},sum(b)={np.sum(ob)}")
  print(f"f/F={np.sum(of)/provider.f_max},b/B={np.sum(ob)/provider.B_max}")
  p_user = users
  if X : p_user = [user for user in users if user.user_id in X]
  for i in p_user:
    print("user ", i.user_id)
    i.print_bounds(p_E,p_N)
    i.print_costs(p_E,p_N)

# =========================== PLOT utils ========================================

def plot_lattice(users, results):
  def all_subsets(n):
    """
    生成从空集到全集的所有子集 (frozenset).
    n: 用户数量
    """
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            yield frozenset(combo)

  def build_subset_graph(n):
      """
      构建包含所有子集的有向图 G.
      节点: 每个子集
      边: 若 A 到 B 差一个元素, A->B
      """
      G = nx.DiGraph()
      # 先把所有子集作为节点加入图
      subsets = list(all_subsets(n))
      for s in subsets:
          G.add_node(s)

      # 对于每个子集, 找只差一个元素的超集
      for s in subsets:
          for i in range(n):
              if i not in s:
                  # s' = s ∪ {i}
                  s_prime = frozenset(s.union({i}))
                  G.add_edge(s, s_prime)

      return G

  n = len(users)  # 用户数量(小于等于10时可视化)
  G = build_subset_graph(n)

  searched_subsets = [node["X"] for node in results]
  searched_edges = []
  for i in range(len(searched_subsets) - 1):
      if searched_subsets[i+1].issuperset(searched_subsets[i]):
          searched_edges.append((searched_subsets[i], searched_subsets[i+1]))

  # ===========可视化============
  # 1) 选一种布局, spring_layout常见, 也可尝试multipartite_layout
  pos = nx.spring_layout(G, seed=42)

  # 2) 画节点和边
  plt.figure(figsize=(10, 8))
  # 节点颜色设置（红色表示搜索过）
  node_colors = ['red' if node in searched_subsets else 'lightblue' for node in G.nodes()]
  nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)

  # 边的颜色设置（橙色表示搜索过的路径）
  edge_colors = ['orange' if edge in searched_edges else 'gray' for edge in G.edges()]
  nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='->')

  # 给每个子集标上标签, 用 str(subset) 作为节点标签
  labels = {s: str(sorted(s)) for s in G.nodes()}
  nx.draw_networkx_labels(G, pos, labels, font_size=10)

  plt.axis('off')
  plt.title(f"All subsets for N={n} (subset lattice)")
  plt.show()

def plot_S_X(users, provider, X, p_E=None, p_N=None, p_E_max=100, p_N_max=100, num_points=200):
  feasible_points = []
  plt.figure(figsize=(8,6))
  if p_E != None and p_N != None: plt.scatter(p_E, p_N)
  p_E_range = np.linspace(1e-3, p_E_max, num_points)
  p_N_range = np.linspace(1e-3, p_N_max, num_points)
  for user in users:
    a, d, b, S, C_l = user.task.alpha, user.task.d, user.task.b, user.S_i, user.cost_local()
    A_i = 2*np.sqrt(a*d)
    B_i = 2*np.sqrt(a*b/S)

    P_E, P_N = np.meshgrid(p_E_range, p_N_range)
    constraint = A_i*np.sqrt(P_E) + B_i*np.sqrt(P_N)

    # 绘制轮廓线 (等于C_l的线)
    CS = plt.contour(P_E, P_N, constraint, levels=[C_l], linewidths=2)
    plt.clabel(CS, inline=True, fontsize=8, fmt={C_l: f'User {user.user_id}'})
    if p_E != None and p_N != None: plt.scatter(p_E, p_N)
  pre_feasible_grid = np.zeros((len(p_E_range), len(p_N_range)), dtype=int)
  for i, pe in enumerate(tqdm(p_E_range, desc="Searching p_E")):
    for j, pn in enumerate(tqdm(p_N_range, desc="Searching p_N")):
      print(pe,pn)
      if models.is_price_pre_feasible(users, provider, X, pe, pn):
          pre_feasible_grid[i, j] = 1
          feasible_points.append((pe, pn))
      else:
        pre_feasible_grid[i, j] = 0

  cmap = ListedColormap(['white', 'lightblue'])
  pre_feasible_grid_T = pre_feasible_grid.T # 在imshow(grid, extent=(xmin,xmax,ymin,ymax)) 中，“grid 的第 0 维”默认对应 y 轴（竖直方向），第 1 维对应 x 轴（水平方向）
  plt.imshow(pre_feasible_grid_T, extent=(p_E_range[0], p_E_range[-1], p_N_range[0], p_N_range[-1]),
            origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, alpha=0.5)
  plt.colorbar(ticks=[0, 1], label='State: 0=Not, 0.5=Pre-Feasible, 1=Feasible')

  plt.xlabel('p_E')
  plt.ylabel('p_N')
  plt.title(f'$S_X$ with {X} in $(p_E, p_N)$ Plane')
  plt.grid()
  plt.show()
  return feasible_points

def plot_P_X(users, provider, X, p_E=None, p_N=None, p_E_max=100, p_N_max=100, num_points=200):
  plt.figure(figsize=(8,6))
  if p_E != None and p_N != None: plt.scatter(p_E, p_N)
  p_E_range = np.linspace(1e-3, p_E_max, num_points)
  p_N_range = np.linspace(1e-3, p_N_max, num_points)
  pre_feasible_grid = np.zeros((len(p_E_range), len(p_N_range)), dtype=float)
  for i, pe in enumerate(tqdm(p_E_range, desc="Searching p_E")):
    for j, pn in enumerate(tqdm(p_N_range, desc="Searching p_N")):
      print(pe,pn)
      if models.is_price_pre_feasible(users, provider, X, pe, pn):
          pre_feasible_grid[i, j] = 0.5
          if models.is_price_feasible(users, provider, X, pe, pn):
            pre_feasible_grid[i, j] = 1
      else:
        pre_feasible_grid[i, j] = 0

  cmap = ListedColormap(['white', 'lightblue', 'red'])
  pre_feasible_grid_T = pre_feasible_grid.T # 在imshow(grid, extent=(xmin,xmax,ymin,ymax)) 中，“grid 的第 0 维”默认对应 y 轴（竖直方向），第 1 维对应 x 轴（水平方向）
  plt.imshow(pre_feasible_grid_T, extent=(p_E_range[0], p_E_range[-1], p_N_range[0], p_N_range[-1]),
            origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, alpha=0.5)
  plt.colorbar(ticks=[0, 0.5, 1], label='State: 0=Not, 0.5=Pre-Feasible, 1=Feasible')

  plt.xlabel('p_E')
  plt.ylabel('p_N')
  plt.title(f'$P_X$ with {X} in $(p_E, p_N)$ Plane')
  plt.grid()
  plt.show()

def plot_user_constraints(users, provider, X, p_E=None, p_N=None, p_E_max=100, p_N_max=100, num_points=200):
    feasible_points, pre_feasible_points = [], []
    plt.figure(figsize=(8,6))

    p_E_range = np.linspace(1e-3, p_E_max, num_points)
    p_N_range = np.linspace(1e-3, p_N_max, num_points)

    # for user in [users[i] for i in X]:
    for user in users:
        a, d, b, S, C_l = user.task.alpha, user.task.d, user.task.b, user.S_i, user.cost_local()
        A_i = 2*np.sqrt(a*d)
        B_i = 2*np.sqrt(a*b/S)

        P_E, P_N = np.meshgrid(p_E_range, p_N_range)
        constraint = A_i*np.sqrt(P_E) + B_i*np.sqrt(P_N)

        # 绘制轮廓线 (等于C_l的线)
        CS = plt.contour(P_E, P_N, constraint, levels=[C_l], linewidths=2)
        plt.clabel(CS, inline=True, fontsize=8, fmt={C_l: f'User {user.user_id}'})
        if p_E != None and p_N != None: plt.scatter(p_E, p_N)

    pre_feasible_grid = np.zeros((len(p_E_range), len(p_N_range)), dtype=int)
    # for i, pe in enumerate(tqdm(p_E_range, desc="Searching p_E")):
    #   for j, pn in enumerate(tqdm(p_N_range, desc="Searching p_N")):
    for i, pe in enumerate(p_E_range):
      for j, pn in enumerate(p_N_range):
        # print(pe,pn)
        if models.is_price_feasible(users, provider, X, pe, pn):
            pre_feasible_grid[i, j] = 2
            feasible_points.append((pe,pn))
        elif models.is_price_pre_feasible(users, provider, X, pe, pn):
            pre_feasible_grid[i, j] = 1
            pre_feasible_points.append((pe,pn))     
        else:
          pre_feasible_grid[i, j] = 0

    cmap = ListedColormap(['white', 'lightblue', 'red'])
    pre_feasible_grid_T = pre_feasible_grid.T # 在imshow(grid, extent=(xmin,xmax,ymin,ymax)) 中，“grid 的第 0 维”默认对应 y 轴（竖直方向），第 1 维对应 x 轴（水平方向）
    plt.imshow(pre_feasible_grid_T, extent=(p_E_range[0], p_E_range[-1], p_N_range[0], p_N_range[-1]),
              origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, alpha=0.5)
    plt.colorbar(ticks=[0, 1, 2], label='State: 0=Not, 0.5=Pre-Feasible, 1=Feasible')

    plt.xlabel('p_E')
    plt.ylabel('p_N')
    plt.title('User Constraints in $(p_E, p_N)$ Plane')
    plt.grid()
    plt.show()
    return feasible_points, pre_feasible_points
