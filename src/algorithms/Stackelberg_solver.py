# src/stackelberg_solver.py
from src.models import User, is_price_feasible, is_price_pre_feasible, ora_solver
from src.algorithms.user_game_solver import greedy_scm
from src.algorithms.baseline_solver import random_solver
import numpy as np
import copy

def branch_and_bound(users, provider, test=False):
    """
    Branch and Bound 算法，用于求解全局的Stackelberg均衡

    参数：
    - users: List[User]，所有用户对象的列表
    - provider: Object<Provider>，provider对象

    返回：
    - X_star: Stackelberg均衡的offloader set
    - U_best: Stackelberg均衡下U_E^*+U_N^*
    """
    # Step 1: 通过Dominance Count构造初始解 (X_init)
    X_init = construct_initial_offloader_set(users)
    print("X_init=",X_init)

    # Step 2: 求解初始解的价格均衡与收益U_best
    p_E_init, p_N_init = incremental_best_response(users, provider, X_init, provider.c_E, provider.c_N)
    f, b, _ = ora_solver([users[i] for i in X_init], provider, p_E_init, p_N_init)
    if f is not None and b is not None: 
      U_best = (p_E_init-provider.c_E)*np.sum(f)+(p_N_init-provider.c_N)*np.sum(b)
    else:
      U_best = 0
      X_init = []
    
    # 初始最优解
    X_star = set(X_init)
    
    # 已探索的集合，初始为空
    Y_init = set()
    
    # 定义递归BnB函数
    def BnB(X, Y, flag, U_parent, p_E_parent, p_N_parent, results=None):
        nonlocal U_best, X_star, X_init

        if results is None: results = []
        
        # Step 6: 计算上界
        _, _, UB_val = node_upper_bound(users, provider, X)
        # _, _, UB_val = upper_bound_sample_frontier_prices(users, provider, X)
        
        # Step 7-9: 上界剪枝
        # if UB_val <= U_best:
        if X != X_init and len(X.union(Y)) != len(users) and UB_val <= U_best:
            node_info = {
              "X": X.copy(),
              "Y": Y.copy(),
              "U_current": None,
              "UB_val": UB_val,
              "U_best": U_best,
              "X_star": X_star.copy(),
              "p_E": None,
              "p_N": None
            }
            results.append(node_info)
            return results

        # Step 10-15: 计算当前收益
        if flag == 'l':
            # 左子树（包含新用户）：重算价格均衡
            p_E, p_N = incremental_best_response(users, provider, X, provider.c_E, provider.c_N)
            # 如果价格不在可行域内，立即剪枝
            # if p_E is None or p_N is None:
            #     print("BnB: ", X, "returns None prices in incremental best resposne.")
            #     node_info = {
            #       "X": X.copy(),
            #       "Y": Y.copy(),
            #       "U_current": None,
            #       "UB_val": UB_val,
            #       "U_best": U_best,
            #       "X_star": X_star.copy(),
            #       "p_E": None,
            #       "p_N": None
            #     }
            #     results.append(node_info)
            #     return results
            f, b, _ = ora_solver([users[i] for i in X], provider, p_E, p_N)
            if f is None or b is None:
              print("BnB: f,b is None.")
              node_info = {
                "X": X.copy(),
                "Y": Y.copy(),
                "U_current": None,
                "UB_val": UB_val,
                "U_best": U_best,
                "X_star": X_star.copy(),
                "p_E": None,
                "p_N": None
              }
              results.append(node_info)
              return results
            U_current = (p_E-provider.c_E)*np.sum(f)+(p_N-provider.c_N)*np.sum(b)
        else:
            # 右子树（排除用户），收益与父节点相同
            U_current = U_parent
            p_E, p_N = p_E_parent, p_N_parent
        
        # Step 16-17: 均衡不可行，剪枝
        if U_current is None:
            print("BnB: U_current is None.")
            node_info = {
              "X": X.copy(),
              "Y": Y.copy(),
              "U_current": None,
              "UB_val": UB_val,
              "U_best": U_best,
              "X_star": X_star.copy(),
              "p_E": None,
              "p_N": None
            }
            results.append(node_info)
            return results
        
        # Step 18-25: 叶子节点检查（已考虑所有用户）
        if len(X.union(Y)) == len(users):
            if U_current > U_best:
                U_best = U_current
                X_star = copy.deepcopy(X)

            node_info = {
              "X": X.copy(),
              "Y": Y.copy(),
              "U_current": U_current,
              "UB_val": UB_val,
              "U_best": U_best,
              "X_star": X_star.copy(),
              "p_E": p_E,
              "p_N": p_N
            }
            results.append(node_info)
            return results
        
        # Step 26: 选择下一个未决策的用户进行分支（简单选择）
        undecided_users = [u.user_id for u in users if u.user_id not in X.union(Y)]
        next_user = undecided_users[0]  # 简单选择第一个未确定用户

        # 记录当前节点
        node_info = {
            "X": X.copy(),  # 记录 X 集合
            "Y": Y.copy(),  # 记录 Y 集合
            "U_current": U_current, # 当前节点的收益
            "UB_val": UB_val, # 当前节点的上界
            "U_best": U_best, # 当前节点处全局的U_best
            "X_star": X_star.copy(), # 当前节点处全局的X_star
            "p_E": p_E,
            "p_N": p_N
        }
        results.append(node_info)

        # Step 27: 左分支：包含用户
        BnB(X.union({next_user}), Y, 'l', U_current, p_E, p_N, results)
        
        # Step 28: 右分支：排除用户
        BnB(X, Y.union({next_user}), 'r', U_current, p_E, p_N, results)

        return results
    
    # Step 31-32: 从初始解开始BnB搜索
    results = BnB(set(X_init), Y_init, 'l', U_best, p_E_init, p_N_init)

    if test: print(results)
    r = next((item for item in results if item["X"] == X_star), None)
    if r is not None: p_E_best, p_N_best = r["p_E"], r["p_N"]
    
    return p_E_best, p_N_best, X_star, U_best, results

# 上层求解器可以扩展更多功能，如利用不同的价格更新策略、传递实验数据等

def incremental_best_response(users, provider, X, initial_p_E, initial_p_N, verbose=False):
    """
    Incremental Best Response 算法，用于求解给定 offloader 集合 X 下的定价均衡。
    
    参数:
      X: offloader 用户的集合（例如：{0, 2, 5}，表示用户ID）
      users: 包含所有用户的列表，每个用户对象需具备属性:
             - user.task.d (计算工作量)
             - user.task.b (数据量)
             - user.task.alpha (延迟敏感度)
      f_max: 系统总计算资源上限
      B_max: 系统总带宽上限
      initial_p_E: ESP 初始价格
      initial_p_N: NSP 初始价格
      c_E: ESP 单位成本（价格下界）
      c_N: NSP 单位成本（价格下界）
      delta: 更新步长
      tolerance: 收敛容差
      max_iter: 最大迭代次数
      eta: 步长收缩因子
      
    返回:
      p_E, p_N: 均衡价格
    """
    max_iter = 2000
    tolerance = 1e-6
    initial_delta_E = 1
    initial_delta_N = 1
    p_E, p_N = initial_p_E, initial_p_N
    delta_E, delta_N = initial_delta_E, initial_delta_N
    eta = 0.3  # 回溯步长因子 (0 < eta < 1)

    for iteration in range(int(max_iter)):
        
        # 更新 p_E，带回溯步长
        p_E_candidate = p_E + delta_E
        while not is_price_feasible(users, provider, X, p_E_candidate, p_N):
            delta_E *= eta
            if abs(delta_E) < tolerance:
                if verbose: print("p_E回溯步长已过小，无法继续寻找可行价格。")
                break
            p_E_candidate = p_E + delta_E
        p_E_new = p_E_candidate if abs(delta_E) >= tolerance else p_E

        # 更新 p_N，带回溯步长
        p_N_candidate = p_N + delta_N
        while not is_price_feasible(users, provider, X, p_E_new, p_N_candidate):
            delta_N *= eta
            if abs(delta_N) < tolerance:
                if verbose: print("p_N回溯步长已过小，无法继续寻找可行价格。")
                break
            p_N_candidate = p_N + delta_N
        p_N_new = p_N_candidate if abs(delta_N) >= tolerance else p_N

        # 确保价格不低于成本
        p_E_new = max(p_E_new, provider.c_E)
        p_N_new = max(p_N_new, provider.c_N)

        # 判断收敛条件（所有delta都小于容差时停止）
        if abs(p_E_new - p_E) < tolerance and abs(p_N_new - p_N) < tolerance:
            if verbose: print(f"收敛于第 {iteration} 次，p_E={p_E_new}, p_N={p_N_new}")
            p_E, p_N = p_E_new, p_N_new
            break

        # 更新价格进入下一次迭代
        p_E, p_N = p_E_new, p_N_new

    else:
        print("达到最大迭代次数，可能未收敛。")

    return p_E, p_N

def node_upper_bound(users, provider, X):
  """
  node_upper_bound 算法，用于求解Branch-and-Bound搜索树中给定节点 (X,Y) 的上界UB(X,Y)。
  
  参数:
    provider: 供应商对象，包含相关信息
    users: 包含所有用户的列表，每个用户对象需具备属性:
            - user.task.d (计算工作量)
            - user.task.b (数据量)
            - user.task.alpha (延迟敏感度)
    X: offloader 用户的集合（例如：{0, 2, 5}，表示用户ID）
  返回:
    best_pE, best_pN: arg max_{p_E,p_N\in S^{front}_X} p_E*f^{max}+p_n*B^{max},
    best_val: UB(X,Y)
  """
  F, B = provider.f_max, provider.B_max
  best_val = -np.inf
  # best_val = np.inf
  best_pE, best_pN = None, None
  
  # 首先，求解每个约束曲线的切点
  for user in [users[i] for i in X]:
      a, d, b, S, C_l = user.task.alpha, user.task.d, user.task.b, user.S_i, user.cost_local()
      
      # 系数定义
      A_i = 2 * np.sqrt(a * d)
      B_i = 2 * np.sqrt(a * b / S)
      
      # 切点解析解
      ratio = (B_i**2 * F) / (A_i**2 * B)
      sqrt_pE = C_l / (A_i + (B_i**2 * F)/(A_i*B))
      p_E_candidate = sqrt_pE**2
      p_N_candidate = ratio * p_E_candidate
      
      # 检查该切点是否满足其他所有用户的约束
      feasible = True
      for other_user in [users[j] for j in X if j != user.user_id]:
          a_o, d_o, b_o, S_o, C_lo = other_user.task.alpha, other_user.task.d, other_user.task.b, other_user.S_i, other_user.cost_local()
          constraint = 2*np.sqrt(a_o*d_o*p_E_candidate) + 2*np.sqrt(a_o*b_o*p_N_candidate/S_o)
          if constraint > C_lo + 1e-6:  # 微小裕量避免数值误差
              feasible = False
              break
      
      if feasible:
          val = (p_E_candidate - provider.c_E)*F + (p_N_candidate - provider.c_N)*B
          if val > best_val:
              best_val = val
              best_pE, best_pN = p_E_candidate, p_N_candidate

  # 如果没有满足约束的单一切点，则搜索交点
  if best_pE is None:
      X_list = list(X)
      for i in range(len(X_list)):
          for j in range(i+1, len(X_list)):
              u1, u2 = users[X_list[i]], users[X_list[j]]
              # 解交点方程组
              try:
                  a1, d1, b1, S1, C1 = u1.task.alpha, u1.task.d, u1.task.b, u1.S_i, u1.cost_local()
                  a2, d2, b2, S2, C2 = u2.task.alpha, u2.task.d, u2.task.b, u2.S_i, u2.cost_local()
                  A = np.array([[2*np.sqrt(a1*d1), 2*np.sqrt(a1*b1/S1)],
                                [2*np.sqrt(a2*d2), 2*np.sqrt(a2*b2/S2)]])
                  C = np.array([C1, C2])
                  sqrt_solution = np.linalg.solve(A, C)
                  p_E_candidate, p_N_candidate = sqrt_solution**2
                  
                  # 验证交点满足所有约束
                  feasible = True
                  for other_user in [users[k] for k in X_list if k not in [u1.user_id, u2.user_id]]:
                      a_o, d_o, b_o, S_o, C_lo = other_user.task.alpha, other_user.task.d, other_user.task.b, other_user.S_i, other_user.cost_local()
                      constraint = 2*np.sqrt(a_o*d_o*p_E_candidate) + 2*np.sqrt(a_o*b_o*p_N_candidate/S_o)
                      if constraint > C_lo + 1e-6:
                          feasible = False
                          break
                  if feasible:
                      val = (p_E_candidate - provider.c_E)*F + (p_N_candidate - provider.c_N)*B
                      if val > best_val:
                          best_val = val
                          best_pE, best_pN = p_E_candidate, p_N_candidate
              except np.linalg.LinAlgError:
                  continue  # 若无解则跳过
                  
  return best_pE, best_pN, best_val


def construct_initial_offloader_set(users, test=False):
    """
    根据论文中“dominance count”的思路，为 Branch-and-Bound 构造初始解 X_init。

    参数:
    ----------
    users : List[User]
        用户列表
    
    返回:
    ----------
    X_init : set
        初始选出的用户下标或 ID 的集合
    """
    # X_init 用于存储初始选出的用户 ID
    # X_init = []
    X_init = set()

    # N 为用户数
    N = len(users)
    # v_i 代表用户指标向量(C^l_i, -\sqrt{alpha_id_i},-\sqrt{\alpha_ib_i/S_i})
    v = []
    for i, user in enumerate(users):
      v.append([user.cost_local(),-1*np.sqrt(user.task.alpha*user.task.d),-1*np.sqrt(user.task.alpha*user.task.b/user.S_i)])
    # R 为指标维度数
    R = 3 if N > 0 else 0

    if test :
      print("v: ", v)


    # 计算 dc_i^+ = sum_{j} 1( v_i^r >= v_j^r )
    #    dc_i^- = sum_{j} 1( v_i^r <= v_j^r )
    # 对每个用户 i, 每个维度 r, 统计 dominance count
    dominance_counts_pos = []  # dominance_counts_pos[i] = [dc_i^1+, dc_i^2+, dc_i^3+]
    dominance_counts_neg = []  # dominance_counts_neg[i] = [dc_i^1-, dc_i^2-, dc_i^3-]

    for i, user_i in enumerate(users):
        dc_i_pos, dc_i_neg = 0, 0
        for j, user_j in enumerate(users):
          if i == j: continue
          if test:
            print("user ", i, " vs. user ", j,": ", v[i], v[j])
            print("   v[i][r] >= v[j][r]:", np.sum([1 for r in range(R) if v[i][r] >= v[j][r]]))
            print("   v[i][r] > v[j][r]:", np.sum([1 for r in range(R) if v[i][r] > v[j][r]]))
            print("   v[i][r] <= v[j][r]:", np.sum([1 for r in range(R) if v[i][r] <= v[j][r]]))
            print("   v[i][r] < v[j][r]:", np.sum([1 for r in range(R) if v[i][r] < v[j][r]]))
          if np.sum([1 for r in range(R) if v[i][r] >= v[j][r]]) == R and \
            np.sum([1 for r in range(R) if v[i][r] > v[j][r]]) > 0:
            if test: print("    dc_",i,"_pos += 1")
            dc_i_pos += 1
          if np.sum([1 for r in range(R) if v[i][r] <= v[j][r]]) == R and \
            np.sum([1 for r in range(R) if v[i][r] < v[j][r]]) > 0:
            if test: print("    dc_",i,"_neg += 1")
            dc_i_neg += 1
        dominance_counts_pos.append(dc_i_pos)
        dominance_counts_neg.append(dc_i_neg)

    if test:
      for i, user_i in enumerate(users):
        print("user ", i, "dc_i+:",dominance_counts_pos[i], "dc_i-:",dominance_counts_neg[i])
      print("dominance_counts_pos:", dominance_counts_pos)
      print("dominance_counts_neg:", dominance_counts_neg)

    # 检查每个用户 i 是否满足 count_r_pos >= 0 for all r
    # 即dc_i^+ > 0
    for i, dc_i_pos in enumerate(dominance_counts_pos):
        # 如果dc_i^+ > 0，那么加入X_init
        if all(dc_i_pos > 0 for r in range(R)):
            X_init.add(i)

    # if X_init == []:
    #   for i, dc_i_neg in enumerate(dominance_counts_neg):
    #     if all(dc_i_neg == 0 for r in range(R)):
    #         X_init.append(i)

    if not X_init:
      s = sorted(users, key=lambda user: - user.cost_local())
      X_init.add(s[0].user_id)

    return X_init



def sample_frontier_prices(users, X, num_points=50, p_E_min=1e-3, p_E_max=100):
    """
    生成前沿集合上的价格采样点。
    """
    frontier_points = []
    for i in X:
        user = users[i]
        alpha, d, b = user.task.alpha, user.task.d, user.task.b
        S_i = user.S_i
        C_l = user.cost_local()

        A = 2 * np.sqrt(alpha * d)
        B = 2 * np.sqrt(alpha * b / S_i)

        sqrt_p_E_vals = np.linspace(np.sqrt(p_E_min), np.sqrt(p_E_max), num_points)

        for sqrt_p_E in sqrt_p_E_vals:
            rhs = C_l - A * sqrt_p_E
            if rhs <= 0:
                continue
            sqrt_p_N = rhs / B
            p_E = sqrt_p_E ** 2
            p_N = sqrt_p_N ** 2
            frontier_points.append((p_E, p_N))

    frontier_points = list(set(frontier_points))
    return frontier_points

def upper_bound_sample_frontier_prices(users, provider, X, num_samples=50):
    """
    给定用户集合 X，返回一个更紧的 UB(X)。
    """
    best_UB = -np.inf
    best_prices = None

    frontier_prices = sample_frontier_prices(users, X, num_samples)

    for p_E, p_N in frontier_prices:
        print(p_E, p_N, best_prices, best_UB)
        # 解 ORA(X) 问题
        users_X = [users[i] for i in X]
        try:
            f_opt, b_opt, success = ora_solver(users_X, provider, p_E, p_N)
            if not success or not f_opt:
                continue  # 求解失败，跳过该价格点
        except Exception as e:
            continue  # 求解失败，跳过

        total_f = np.sum(f_opt)
        total_b = np.sum(b_opt)

        # 计算当前价格组合下的收益UB
        UB = (p_E - provider.c_E) * total_f + (p_N - provider.c_N) * total_b

        # 更新UB
        if UB > best_UB:
            print(UB, best_UB)
            best_UB = UB
            best_prices = (p_E, p_N)

    if best_prices is None:
        print("No feasible price points found on frontier.")
        return None, None, None

    return best_prices[0], best_prices[1], best_UB
    
