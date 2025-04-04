import src.config as config
import src.models as models
import src.algorithms.user_game_solver as user_game_solver
import src.algorithms.Stackelberg_solver as Stackelberg_solver
import src.run_experiment as run_experiment

import numpy as np
import itertools
from scipy.optimize import minimize

def epf_baseline(users, provider, initial_p_E, initial_p_N, 
                 lambda_step=0.01, tol=1e-3, max_iter=1000):
    """
    EPF Baseline：基于 Jiadi Liu 思路的 Equilibrium Price Finding 算法。
    
    该算法采用类似 tatonnement 的方法调整价格，使得
    offloader 集合 X 下的真实资源需求（通过 ORA 求解）与系统总资源 (F, B) 达到平衡，
    从而求得 Stackelberg 均衡价格。
    
    参数:
      - users: 用户对象列表，每个用户包含任务参数、local_cpu、cost_local() 等。
      - provider: Provider 对象，包含 f_max, B_max, c_E, c_N 等参数。
      - initial_p_E, initial_p_N: 初始价格（建议选高于成本价，如 provider.c_E+δ, provider.c_N+δ）。
      - lambda_step: 调价步长因子，控制价格更新幅度。
      - tol: 收敛容差，当价格更新幅度低于 tol 时认为收敛。
      - max_iter: 最大迭代次数。
      
    返回:
      - p_E, p_N: 均衡价格
      - X_final: 对应的 offloader 用户集合（由 greedy_scm 得到）
      - utility: 当前价格下的U_E+U_N
      - history: 每轮记录一个字典，包含 p_E, p_N, sum_f, sum_b, 以及联合收益
    """
    p_E, p_N = initial_p_E, initial_p_N
    history = []
    utility = -np.inf
    
    for it in range(max_iter):
        # 用当前价格求 offloader 集合 X（由用户博弈求均衡）
        X = user_game_solver.greedy_scm(users, provider, p_E, p_N)
        if not X or len(X)==0:
            # 如果没有用户 offload，则直接结束，返回当前价格和空集合
            history.append({
                "iteration": it,
                "p_E": p_E,
                "p_N": p_N,
                "X": X,
                "sum_f": 0,
                "sum_b": 0,
                "utility": 0
            })
            utility = 0
            break

        # 针对 offloader 集合 X 求解资源分配（ORA）
        users_subset = [users[i] for i in X]
        f, b, success = models.ora_solver(users_subset, provider, p_E, p_N)
        if not success or f is None or b is None:
            # 如果 ORA 求解失败，则略过本轮（也可考虑调低价格）
            history.append({
                "iteration": it,
                "p_E": p_E,
                "p_N": p_N,
                "X": X,
                "sum_f": None,
                "sum_b": None,
                "utility": -np.inf
            })
            utility = -np.inf
            # 可以选择调整价格或继续下次迭代，此处直接 break
            break

        sum_f = np.sum(f)
        sum_b = np.sum(b)
        # 计算当前联合收益（仅作为参考）
        utility = (p_E - provider.c_E)*sum_f + (p_N - provider.c_N)*sum_b
        
        # 记录当前状态
        history.append({
            "iteration": it,
            "p_E": p_E,
            "p_N": p_N,
            "X": X.copy(),
            "sum_f": sum_f,
            "sum_b": sum_b,
            "utility": utility
        })
        
        # 计算超额需求：需求与供应之差
        excess_f = sum_f - provider.f_max
        excess_b = sum_b - provider.B_max
        
        # 更新价格，根据超额需求调整价格：
        # 如果需求大于供应，excess > 0，则价格上调；
        # 如果需求小于供应，excess < 0，则价格下调。
        p_E_new = p_E + lambda_step * (excess_f / provider.f_max)
        p_N_new = p_N + lambda_step * (excess_b / provider.B_max)
        
        # 如果价格更新幅度小于容差，则认为收敛
        if abs(p_E_new - p_E) < tol and abs(p_N_new - p_N) < tol:
            p_E, p_N = p_E_new, p_N_new
            break
        
        p_E, p_N = p_E_new, p_N_new
    
    # 最终 offloader 集合
    X_final = user_game_solver.greedy_scm(users, provider, p_E, p_N)
    return p_E, p_N, X_final, utility, history


def tutuncuoglu_exhaustive_search(users, provider, verbose=False):
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
  p_best = []
  results = []
  for subset in all_subsets(I):
      if not subset:
          continue
      if verbose: print("Subset: ", subset)
      user_subset = [u for u in users if u.user_id in subset]
      f, b, c = optimize_allocation_delay(user_subset, provider)
      if f is None or b is None:
        U_current = 0
        p = np.zeros(len(user_subset))
        sum_f,sum_b = 0,0
      else:
        p = [u.cost_local()-u.task.alpha*u.offloading_time(f[i],b[i]) for i, u in enumerate(user_subset)]
        U_current = np.sum(p)
        sum_f, sum_b = np.sum(f), np.sum(b)
      if verbose: 
        print(" U^*: ", U_current)
        print("p:", p)
        print("c:", c)
        print("f:", f, "b:", b)

      results.append({
          "X": subset,
          "U_X": U_current,
          "p": list(p),
          "sum_f": sum_f,
          "sum_b": sum_b,
          "f": f,
          "b": b
      })

      if U_current > U_best_exhau:
          U_best_exhau = U_current
          X_best_exhau = subset
          p_best = p

  if verbose: print("Best subset: ", X_best_exhau)
  if verbose: print("Best value: ", U_best_exhau)
  return X_best_exhau, U_best_exhau, p_best, results


def optimize_allocation_delay(offloaders, provider, verbose=False):
    """
    用 scipy.optimize 求解用户加权卸载时延最小化，并在给定的约束下找到最优的资源分配。

    参数：
    users: 用户列表，假设每个用户对象有属性 w (计算任务), d (数据量)
    provider
    返回：
    结果字典，包含最优的 f_i 和 b_i 资源分配、目标函数值和其他信息。
    """

    # 用户数量
    n = len(offloaders)

    alpha = [u.task.alpha for u in offloaders]
    d = [u.task.d for u in offloaders]
    b = [u.task.b for u in offloaders]
    S = [u.S_i for u in offloaders]
    f_lower_bound_array = np.array([user.local_cpu for user in offloaders])
    cost_local_array = np.array([user.cost_local() for user in offloaders])

    # 初始化 f_i 和 b_i
    f_init = np.array(f_lower_bound_array) + 1e-3  # 假设初始化为一个小的值
    B_init = np.ones(n) + 1e-3

    # 定义目标函数
    def objective(x):
        f, B = x[:n], x[n:]
        delay_cost = np.sum(alpha * (b / (B * S) + d / f))

        return delay_cost

    # 定义约束条件
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[:n] - f_lower_bound_array},        # f_i >= f_lower_bound_array
        {'type': 'ineq', 'fun': lambda x: x[n:] - 1e-6},                       # B_i >= 1e-6
        {'type': 'ineq', 'fun': lambda x: provider.f_max - np.sum(x[:n])},     # 总计算资源限制
        {'type': 'ineq', 'fun': lambda x: provider.B_max - np.sum(x[n:])},     # 总带宽限制
    ]

    
    # 添加新的约束：对于每个用户 i，
    #    d_i / f_i + b_i/(B_i * S_i) - d_i / f^l_i <= 0
    for i in range(n):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: d[i] / f_lower_bound_array[i] - d[i] / x[i] + b[i] / (x[n+i] * S[i])
        })

    # 变量上下界约束（Bounds）
    bounds = [(f_lower_bound_array[i], provider.f_max) for i in range(n)] + \
             [(1e-6, provider.B_max) for _ in range(n)]

    # 设置初始猜测的 f_i 和 b_i
    x0 = np.concatenate([f_init, B_init])

    # 使用 scipy.optimize.minimize 求解优化问题
    result = minimize(
        lambda x: objective(x),
        x0, method='SLSQP',
        constraints=cons,
        bounds=bounds,
        options={'disp': verbose, 'ftol': 1e-6, 'maxiter': 1000}
    )

    # 提取优化结果
    f_opt = result.x[:n]
    b_opt = result.x[n:]

    # 计算最优目标函数值
    C_opt = np.sum(alpha * (d / f_opt + b / (S * b_opt)))

    return f_opt, b_opt, C_opt