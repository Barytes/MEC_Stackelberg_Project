# src/baseline_solver.py
"""
在此模块中实现其他 baseline 算法，例如随机策略、局部搜索等。
接口要求与 greedy_solver 保持一致，便于在 run_experiment.py 中统一调用。
"""
import src.config as config
import src.models as models
import src.algorithms.user_game_solver as user_game_solver
import src.algorithms.Stackelberg_solver as Stackelberg_solver
import src.run_experiment as run_experiment

import numpy as np
import itertools
from tqdm import tqdm
from itertools import product

def oracle_baseline_exhau_gne(users, provider, p_E_range=(0.1,100.1), p_N_range=(0.1,100.1), grid_points=200):
    """
    Oracle baseline 算法：
      在给定的价格范围内对 p_E, p_N 进行网格搜索，
      对于每个价格点，用 greedy_scm 求解用户均衡 X，
      再用 ORA 求解资源分配，计算联合收益：
         Revenue = (p_E - c_E) * sum(f) + (p_N - c_N) * sum(b)
      最后返回收益最大的价格组合、对应的用户集合以及收益值。

    参数:
        users: 用户列表
        provider: Provider 对象，须包含 c_E, c_N 属性
        p_E_range: (min, max) 的 p_E 范围
        p_N_range: (min, max) 的 p_N 范围
        grid_points: 网格搜索的分辨率（每个维度的点数）

    返回:
        best_p_E, best_p_N: 收益最大的价格
        best_X: 对应的用户均衡 offloader 集合（例如用户 id 的集合）
        best_revenue: 最大联合收益
    """
    best_revenue = -np.inf
    best_p_E, best_p_N = None, None
    best_X = None
    results = []

    p_E_vals = np.linspace(p_E_range[0], p_E_range[1], grid_points)
    p_N_vals = np.linspace(p_N_range[0], p_N_range[1], grid_points)

    # for i, p_E in enumerate(tqdm(p_E_vals, desc="Searching p_E")):
    #   for j, p_N in enumerate(tqdm(p_N_vals, desc="Searching p_N")):
    for i, p_E in enumerate(p_E_vals):
      for j, p_N in enumerate(p_N_vals):
            non_pre_feasible_counter = np.sum([1 for u in users if u.C_hat_e_i(p_E,p_N)>=u.cost_local()+1e-8])
            if non_pre_feasible_counter == len(users):
              results.append({
                  "X": X,
                  "U_X": 0,
                  "p_E": p_E,
                  "p_N": p_N,
                  "sum_f": 0,
                  "sum_b": 0
              })
              continue
            
            U_current = 0
            # 计算给定价格下的用户均衡 offloader 集合 X
            # X = user_game_solver.greedy_scm(users, provider, p_E, p_N)
            X, oc, f, b = exhaustive_gne(users, provider, p_E, p_N)
            # print(f"X={X}")
            if not X or len(X) == 0:
                results.append({
                    "X": X,
                    "U_X": 0,
                    "p_E": p_E,
                    "p_N": p_N,
                    "sum_f": 0,
                    "sum_b": 0
                })
                continue
            else:
              user_subset = [users[i] for i in X]
              # print(f"f={f},b={b}")
              if f is None or b is None: 
                results.append({
                  "X": X,
                  "U_X": 0,
                  "p_E": p_E,
                  "p_N": p_N,
                  "sum_f": 0,
                  "sum_b": 0
                })
                continue
              else:
                U_current = (p_E-provider.c_E)*np.sum(f)+(p_N-provider.c_N)*np.sum(b)
                # print("U_current=", U_current)
                results.append({
                    "X": X,
                    "U_X": U_current,
                    "p_E": p_E,
                    "p_N": p_N,
                    "sum_f": np.sum(f),
                    "sum_b": np.sum(b)
                })

            # 计算联合收益：ESP 和 NSP 的收益之和
            if U_current > best_revenue:
                best_revenue = U_current
                best_p_E, best_p_N = p_E, p_N
                best_X = X.copy()

    return best_p_E, best_p_N, best_X, best_revenue, results

def oracle_baseline_greedy(users, provider, p_E_range=(0.1,100.1), p_N_range=(0.1,100.1), grid_points=200):
    """
    Oracle baseline 算法：
      在给定的价格范围内对 p_E, p_N 进行网格搜索，
      对于每个价格点，用 greedy_scm 求解用户均衡 X，
      再用 ORA 求解资源分配，计算联合收益：
         Revenue = (p_E - c_E) * sum(f) + (p_N - c_N) * sum(b)
      最后返回收益最大的价格组合、对应的用户集合以及收益值。

    参数:
        users: 用户列表
        provider: Provider 对象，须包含 c_E, c_N 属性
        p_E_range: (min, max) 的 p_E 范围
        p_N_range: (min, max) 的 p_N 范围
        grid_points: 网格搜索的分辨率（每个维度的点数）

    返回:
        best_p_E, best_p_N: 收益最大的价格
        best_X: 对应的用户均衡 offloader 集合（例如用户 id 的集合）
        best_revenue: 最大联合收益
    """
    best_revenue = -np.inf
    best_p_E, best_p_N = None, None
    best_X = None
    results = []

    p_E_vals = np.linspace(p_E_range[0], p_E_range[1], grid_points)
    p_N_vals = np.linspace(p_N_range[0], p_N_range[1], grid_points)

    # for i, p_E in enumerate(tqdm(p_E_vals, desc="Searching p_E")):
    #   for j, p_N in enumerate(tqdm(p_N_vals, desc="Searching p_N")):
    for i, p_E in enumerate(p_E_vals):
      for j, p_N in enumerate(p_N_vals):
            non_pre_feasible_counter = np.sum([1 for u in users if u.C_hat_e_i(p_E,p_N)>=u.cost_local()+1e-8])
            if non_pre_feasible_counter == len(users):
              results.append({
                  "X": X,
                  "U_X": 0,
                  "p_E": p_E,
                  "p_N": p_N,
                  "sum_f": 0,
                  "sum_b": 0
              })
              continue
            
            U_current = 0
            # 计算给定价格下的用户均衡 offloader 集合 X
            X = user_game_solver.greedy_scm(users, provider, p_E, p_N)
            # print(f"X={X}")
            if not X or len(X) == 0:
                results.append({
                    "X": X,
                    "U_X": 0,
                    "p_E": p_E,
                    "p_N": p_N,
                    "sum_f": 0,
                    "sum_b": 0
                })
            else:
              user_subset = [users[i] for i in X]
              f, b, _ = models.ora_solver(user_subset, provider, p_E, p_N) # 求解 ORA 得到真实资源分配 (f, b)
              # print(f"f={f},b={b}")
              if f is None or b is None: 
                results.append({
                  "X": X,
                  "U_X": 0,
                  "p_E": p_E,
                  "p_N": p_N,
                  "sum_f": 0,
                  "sum_b": 0
                })
              else:
                U_current = (p_E-provider.c_E)*np.sum(f)+(p_N-provider.c_N)*np.sum(b)
                # print("U_current=", U_current)
                results.append({
                    "X": X,
                    "U_X": U_current,
                    "p_E": p_E,
                    "p_N": p_N,
                    "sum_f": np.sum(f),
                    "sum_b": np.sum(b)
                })

            # 计算联合收益：ESP 和 NSP 的收益之和
            if U_current > best_revenue:
                best_revenue = U_current
                best_p_E, best_p_N = p_E, p_N
                best_X = X.copy()

    return best_p_E, best_p_N, best_X, best_revenue, results

def is_nash(X, users, provider, p_E, p_N):
    # 检查每个用户单独偏离是否会改善成本
    for i in range(len(users)):
        if i in X:  # offloader用户，看是否local更优
            new_X = X - {i}
        else:       # local用户，看是否offload更优
            new_X = X | {i}
        f_new, b_new, cost_new = models.ora_solver([users[j] for j in new_X], provider, p_E, p_N)
        if f_new is None: continue
        # 原成本
        f_orig, b_orig, cost_orig = models.ora_solver([users[j] for j in X], provider, p_E, p_N)
        if cost_new < cost_orig:
            return False  # 单方面偏离有利可图，不是Nash
    return True

def exhaustive_gne(users, provider, p_E, p_N):
    # best_X, best_cost = None, np.inf
    best_X, best_rev = None, np.inf
    best_f, best_b = 0, 0
    for decision_vector in product([0,1], repeat=len(users)):
        X = {i for i, decision in enumerate(decision_vector) if decision == 1}
        if not X: continue
        f, b, cost = models.ora_solver([users[i] for i in X], provider, p_E, p_N)
        if f is None: continue  # 无效集合
        if is_nash(X, users, provider, p_E, p_N):
            rev = (p_E-provider.c_E)*np.sum(f)+(p_N-provider.c_N)*np.sum(b)
            if rev > best_rev:
                best_rev, best_X = rev, X
                best_f, best_b = f, b
    return best_X, best_rev, best_f, best_b

def random_offloader_baseline(users, provider, num_trials=100):
    """
    随机选择 offloader 集合，利用增量最佳响应求解价格，
    并计算联合收益。重复多次取收益最高的解作为baseline结果。
    
    参数:
      users: 用户对象列表，用户对象需包含 user_id、task (含 d, b, alpha)、
             local_cpu，以及 cost_local() 方法。
      provider: Provider 对象，包含 f_max, B_max, c_E, c_N 等参数。
      num_trials: 随机试验次数
      
    返回:
      best_X: 收益最高的 offloader 集合（集合内存储用户id）
      best_price: 对应的价格 (p_E, p_N)
      best_utility: 最高联合收益
      log_results: 每次试验的记录列表（包含 X, p_E, p_N, utility 等）
    """
    best_utility = -np.inf
    best_X, best_price = None, None
    log_results = []
    N = len(users)
    user_ids = list(range(N))
    
    for trial in range(num_trials):
        # 随机选择 offloader 集合，随机选择集合大小（至少1个）
        size = np.random.randint(1, N)
        X = set(np.random.choice(user_ids, size, replace=False))
        
        # 利用增量最佳响应求解均衡价格
        # 这里假设 incremental_best_response 已经实现，
        # 它返回 (p_E, p_N)；注意初始价格不要选成本价而是稍高一些
        p_E, p_N = Stackelberg_solver.incremental_best_response(users, provider, X, provider.c_E + 1e-3, provider.c_N + 1e-3)
        
        # 若价格求解失败，跳过此试验
        if p_E is None or p_N is None:
            continue

        # 用 ORA 求解真实资源分配
        users_X = [users[i] for i in X]
        f, b, success = models.ora_solver(users_X, provider, p_E, p_N)
        if not success or f is None or b is None:
            continue
        
        utility = (p_E - provider.c_E) * np.sum(f) + (p_N - provider.c_N) * np.sum(b)
        log_results.append({
            "X": X.copy(),
            "p_E": p_E,
            "p_N": p_N,
            "utility": utility,
            "sum_f": np.sum(f),
            "sum_b": np.sum(b)
        })
        
        if utility > best_utility:
            best_utility = utility
            best_X = X.copy()
            best_price = (p_E, p_N)
    
    return best_X, best_price, best_utility, log_results

def threshold_offloader_set(users, provider):
    """
    根据阈值规则构造 offloader 集合：
    threshold为用户的平均本地成本
    若对于用户i，其本地成本大于 threshold，则选择用户卸载。
      
    返回：
      X: offloader 用户集合（用户id的集合）
    """
    X = set()
    # threshold = np.array([u.cost_local() for u in users]).mean()
    # for user in users:
    #     c_local = user.cost_local()
    #     # 估计卸载成本
    #     if c_local > threshold:
    #         X.add(user.user_id)
    threshold = np.array([u.task.alpha for u in users]).mean()
    for user in users:
        if user.task.alpha > threshold:
            X.add(user.user_id)
    return X

def threshold_based_offloader_baseline(users, provider, initial_p_E=None, initial_p_N=None):
    """
    基于阈值规则构造 offloader 集合，然后利用增量最佳响应求解 Stackelberg 均衡。
    
    参数:
      threshold: 用于offloader选择的阈值
      如果 initial_p_E, initial_p_N 未指定，则使用成本价+一个小正数
      
    返回:
      X: offloader 集合
      (p_E, p_N): 均衡价格
      utility: 联合收益
      log_info: 包含 X, p_E, p_N, utility 的记录字典
    """
    if initial_p_E is None:
        initial_p_E = provider.c_E + 1e-3
    if initial_p_N is None:
        initial_p_N = provider.c_N + 1e-3
        
    X = threshold_offloader_set(users, provider)
    if len(X) == 0:
        return None, None, None, None  # 无 offloader
    
    # 求解均衡价格
    p_E, p_N = Stackelberg_solver.incremental_best_response(users, provider, X, initial_p_E, initial_p_N)
    if p_E is None or p_N is None:
        return None, None, None, None
    
    # 求解真实资源分配
    users_X = [users[i] for i in X]
    f, b, success = models.ora_solver(users_X, provider, p_E, p_N)
    if not success or f is None or b is None:
        return None, None, None, None

    utility = (p_E - provider.c_E) * np.sum(f) + (p_N - provider.c_N) * np.sum(b)
    log_info = {
        "X": X.copy(),
        "p_E": p_E,
        "p_N": p_N,
        "utility": utility,
        "sum_f": np.sum(f),
        "sum_b": np.sum(b)
    }
    return X, (p_E, p_N), utility, log_info


def exhaustive_search(users, provider, verbose=False):
  def all_subsets(s):
      """
      返回集合 s 的所有子集（以元组形式）
      """
      s = list(s)
      return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

  # 示例：遍历集合 {1, 2, 3} 的所有子集
  I = range(len(users))
  U_best_exhau = -np.inf
  X_best_exhau = []
  results = []
  for subset in all_subsets(I):
      if not subset:
          continue
      if verbose: print("Subset: ", subset)
      p_E, p_N = Stackelberg_solver.incremental_best_response(users, provider, subset, provider.c_E, provider.c_N)
      user_subset = [u for u in users if u.user_id in subset]
      if p_E is None or p_N is None:
          continue
      f, b, _ = models.ora_solver(user_subset, provider, p_E, p_N)
      if f is None or b is None:
        U_current = 0
      else:
        U_current = (p_E-provider.c_E)*np.sum(f)+(p_N-provider.c_N)*np.sum(b)
      if verbose: print(" U^*: ", U_current)

      results.append({
          "X": subset,
          "U_X": U_current,
          "p_E": p_E,
          "p_N": p_N,
          "sum_f": np.sum(f),
          "sum_b": np.sum(b)
      })

      if U_current > U_best_exhau:
          U_best_exhau = U_current
          X_best_exhau = subset

  if verbose: print("Best subset: ", X_best_exhau)
  if verbose: print("Best value: ", U_best_exhau)
  return X_best_exhau, U_best_exhau, results


def best_response_update(users, provider, 
            initial_p_E, initial_p_N, 
            step_p=1e-3, max_iter=1000,
            verbose=False):
    """
    DO NOT USE
    这个算法的逻辑不是很合理，看起来像是重做了一遍IBR。
    Best Response Update (Backward Induction) Algorithm.
    
    参数:
        - users: 所有用户列表
        - provider: Provider 对象
        - initial_p_E, initial_p_N: ESP与NSP的初始价格
        - step_p: 每次迭代增加的价格步长
        - max_iter: 最大迭代次数，防止无限循环
    
    返回:
        - Stackelberg均衡价格 (p_E, p_N)
        - 对应的用户集合X_star
        - 对应的收益(U_E, U_N)
    """
    p_E, p_N = initial_p_E, initial_p_N
    iter_count = 0
    
    # 初始状态，计算初始效用
    U_E_current, U_N_current = 0, 0
    X_star = user_game_solver.greedy_scm(users, provider, p_E, p_N)
    if X_star is None: 
      print("Best Response Update: Configuration infeasibel, initial X is empty, nobody offload.")
    f, b, oc = models.ora_solver([users[i] for i in X_star], provider, p_E, p_N)
    if oc is not None:
      U_E_current = (p_E-provider.c_E)*np.sum(f)
      U_N_current = (p_N-provider.c_N)*np.sum(b)

    history = [{
      "iteration": iter_count,
      "p_E": p_E,
      "p_N": p_N,
      "X": X_star.copy(),
      "U_E": U_E_current,
      "U_N": U_N_current
    }]
    
    # 标记是否可继续更新
    can_update_E, can_update_N = True, True
    
    while (can_update_E or can_update_N) and iter_count < max_iter:
        iter_count += 1
        # 记录更新前的价格和收益
        p_E_prev, p_N_prev = p_E, p_N
        U_E_prev, U_N_prev = U_E_current, U_N_current
        X_prev = X_star
        
        #  ESP 尝试增大价格
        if can_update_E:
            p_E_candidate = p_E + step_p
            X_candidate = user_game_solver.greedy_scm(users, provider, p_E_candidate, p_N)
            f_candidate, b_candidate, _ = models.ora_solver([users[i] for i in X_candidate], provider, p_E_candidate, p_N)
            if f_candidate is None or b_candidate is None:
                U_E_candidate = -np.inf  # 显式表示不可行，BRU 不会再考虑这个价格
            else:
                U_E_candidate = (p_E_candidate-provider.c_E)*np.sum(f_candidate)
            
            if U_E_candidate > U_E_current:
                p_E, U_E_current, X_star = p_E_candidate, U_E_candidate, X_candidate
                if verbose: print(f"ESP 增大价格至 {p_E:.4f}，收益变为 {U_E_current:.4f}，X: {X_star}")
            else:
                can_update_E = False  # 无法继续增大
            
        #  NSP 尝试增大价格
        if can_update_N:
            p_N_candidate = p_N + step_p
            X_candidate = user_game_solver.greedy_scm(users, provider, p_E, p_N_candidate)
            f_candidate, b_candidate, _ = models.ora_solver([users[i] for i in X_candidate], provider, p_E, p_N_candidate)
            if f_candidate is None or b_candidate is None:
              U_N_candidate = -np.inf
            else:
              U_N_candidate = (p_N_candidate-provider.c_N)*np.sum(b_candidate)
            
            if U_N_candidate > U_N_current:
                p_N, U_N_current, X_star = p_N_candidate, U_N_candidate, X_candidate
                if verbose: print(f"NSP 增大价格至 {p_N:.4f}，收益变为 {U_N_current:.4f}，X: {X_star}")
            else:
                can_update_N = False  # 无法继续增大

        # 每轮都记录历史信息
        history.append({
            "iteration": iter_count,
            "p_E": p_E,
            "p_N": p_N,
            "X": X_star.copy(),
            "U_E": U_E_current,
            "U_N": U_N_current
        })
        
        # 若两边均未更新价格，跳出循环
        if p_E == p_E_prev and p_N == p_N_prev:
            break


    # 最终均衡情况
    f_final, b_final, _ = models.ora_solver([users[i] for i in X_star], provider, p_E, p_N)
    U_E_final = (p_E-provider.c_E)*np.sum(f_final)
    U_N_final = (p_N-provider.c_N)*np.sum(b_final)
    print(f"均衡找到，迭代次数: {iter_count}，最终价格：p_E={p_E:.4f}, p_N={p_N:.4f}")
    print(f"最终收益：ESP={U_E_final:.4f}, NSP={U_N_final:.4f}")
    
    return p_E, p_N, X_star, U_E_final, U_N_final, history


def random_solver(users, p_E, p_N):
    """
    随机选择 offload 决策，并随机分配资源
    """
    import numpy as np
    for user in users:
        user.offload = np.random.choice([0, 1])
        if user.offload == 1:
            # 随机生成资源分配（需要满足资源约束，可进一步优化）
            user.f_i = np.random.uniform(0.1, np.sqrt(user.task.alpha * user.task.d / p_E))
            user.B_i = np.random.uniform(0.1, np.sqrt(user.task.alpha * user.task.b / p_N))
        else:
            user.f_i, user.B_i = 0.0, 0.0
    return users

# 可添加其他 baseline 算法，并提供统一接口
