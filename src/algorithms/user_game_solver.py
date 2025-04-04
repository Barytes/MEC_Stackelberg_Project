# src/greedy_solver.py
import numpy as np
from src.models import is_offloader_set_feasible, ora_solver, social_cost

def greedy_scm(users, provider, p_E, p_N, verbose=False):
  """
    贪心算法（Algorithm 1），近似求解 SCM 问题中的用户选择集合。
    
    参数：
    - users: 所有用户的可迭代对象，假设 user.user_id 是其标识
    - provider: 供应商对象
    - p_E, p_N: 给定价格
    
    返回：
    - X_star: 最终贪心选出的用户集合（以 user_id 的 set 形式）
    """
  # 1) 初始化
  # X = []   # 当前选中用户集合，user id
  X = set()
  s = sorted(users, key=lambda user: user.C_hat_e_i(p_E, p_N) - user.cost_local())
  # for u in s:
  #   if u.cost_local() <= u.C_hat_e_i(p_E,p_N)-1e-8:
  #     continue
  #   X_u = X.copy()  # 或者：X_copy = set(X)
  #   X_u.add(u.user_id)
  #   if np.sum([user.f_hat(p_E) for user in users if user.user_id in X_u]) <= provider.f_max \
  #     and np.sum([user.B_hat(p_N) for user in users if user.user_id in X_u]) <= provider.B_max:
  #     X.add(u.user_id)

  if verbose: print(f"Greedy: X_0={X}")
  
  # 2) while X^t is feasible
  #    这里假设 "可行" 指的是当前 X 自身可行
  while is_offloader_set_feasible(users, provider, X, p_E, p_N):
      # 计算 offloader set = X时的social cost V(X)
      if verbose: print("X:", X)
      user_X = []
      if X :
        user_X = [u for u in users if u.user_id in X]
        f_X,b_X,Ce_X = ora_solver(user_X, provider, p_E, p_N)
        V_X = Ce_X + np.sum([u.cost_local() for u in users if u.user_id not in X])
        if verbose: print(" V_X=",V_X)
        if V_X < 0: 
          if verbose: print(f"X={X}, Ce_X={Ce_X}")
          break
      else:
        V_X = np.sum([u.cost_local() for u in users])

      # 3) 在 I \ X 中找到增量成本最小的用户 i*
      candidate_users = [u for u in s if u.user_id not in X]
      if not candidate_users:
          # 没有剩余用户可选，直接跳出
          break

      # 计算每个候选用户的 ∆V(X, {i})
      min_delta_v = float('inf')
      best_i = None
      for u in candidate_users:
        if verbose: print("candidate user:", u.user_id)
        f_Xu,b_Xu,Ce_Xu = ora_solver(user_X+[u], provider, p_E, p_N)
        if Ce_Xu is None: continue
        Xu = X.copy()
        Xu.add(u.user_id)
        V_Xu = Ce_Xu + np.sum([j.cost_local() for j in users if j.user_id not in Xu])
        if V_Xu < 0:
          if verbose: print(f"X={X}, u={u}, Ce_Xu={Ce_Xu}")
          continue
        dv = V_Xu - V_X
        if verbose: print(" V_Xu=",V_Xu, ", dv=",dv)
        if dv < min_delta_v:
            min_delta_v = dv
            best_i = u.user_id

      # 4) 如果 ∆V(X, {i*}) <= 0, 则将 i* 加入 X
      if min_delta_v <= 0:
          if verbose: print(f"best u={best_i} at X={X}")
          X.add(best_i)
      else:
          # 如果最小的增量成本都 > 0，说明再加入任何用户都会增加总成本，停止
          if not X:
            print("Greedy SCM: Configuration Infeasible, nobody offloads.")
          break

  # 9) return X^t 作为贪心解
  return X

# 如果需要，可添加更多贪心更新策略、约束判断等
